// Package hnsw implements a Hierarchical Navigable Small World graph for
// approximate nearest-neighbour search, built from scratch on the Go standard
// library.
//
// HNSW organises points into a stack of proximity graphs. The bottom layer
// (level 0) contains every point; each higher layer contains an
// exponentially thinning random subset. A search starts at a single entry
// point in the top layer, greedily walks toward the query, then descends a
// level and repeats. The upper layers act like an express highway that gets
// the search close to the target in O(log n) hops before the dense bottom
// layer refines the result. This is the same structure used by production
// engines such as FAISS-HNSW, Milvus and Weaviate; here it is implemented by
// hand to show how it works.
package hnsw

import (
	"container/heap"
	"errors"
	"math"
	"math/rand"
	"sync"

	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

// Common errors.
var (
	ErrDimMismatch  = errors.New("hnsw: vector dimension mismatch")
	ErrEmptyVector  = errors.New("hnsw: empty vector")
	ErrNotFound     = errors.New("hnsw: id not found")
)

// Params configures an index. Zero values fall back to sensible defaults via
// NewIndex.
type Params struct {
	Dim            int          // vector dimensionality (required)
	Metric         vmath.Metric // distance metric
	M              int          // max neighbours per node on layers > 0
	EfConstruction int          // candidate list size while building
	EfSearch       int          // candidate list size while querying
	Seed           int64        // RNG seed for reproducible level assignment
}

// node is a single point and its per-layer adjacency lists. links[l] holds the
// neighbour ids on layer l; len(links) == node level + 1.
type node struct {
	id      uint32
	vec     []float32
	links   [][]uint32
	deleted bool
}

// Index is a thread-safe HNSW graph. Reads (Search) take a read lock; writes
// (Add, Delete) take the write lock.
type Index struct {
	mu sync.RWMutex

	dim        int
	metric     vmath.Metric
	m          int // M: neighbours on upper layers
	mMax0      int // max neighbours on layer 0 (2*M by convention)
	efConstr   int
	efSearch   int
	levelMult  float64 // 1/ln(M), controls layer assignment

	nodes      map[uint32]*node
	entryPoint uint32
	maxLevel   int
	hasEntry   bool

	rng *rand.Rand
}

// NewIndex builds an empty index, applying defaults for any zero parameter.
func NewIndex(p Params) (*Index, error) {
	if p.Dim <= 0 {
		return nil, errors.New("hnsw: Dim must be > 0")
	}
	if p.M <= 0 {
		p.M = 16
	}
	if p.EfConstruction <= 0 {
		p.EfConstruction = 200
	}
	if p.EfSearch <= 0 {
		p.EfSearch = 64
	}
	if p.Seed == 0 {
		p.Seed = 42
	}
	return &Index{
		dim:       p.Dim,
		metric:    p.Metric,
		m:         p.M,
		mMax0:     p.M * 2,
		efConstr:  p.EfConstruction,
		efSearch:  p.EfSearch,
		levelMult: 1 / math.Log(float64(p.M)),
		nodes:     make(map[uint32]*node),
		rng:       rand.New(rand.NewSource(p.Seed)),
	}, nil
}

// Len returns the number of live (non-deleted) vectors.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	n := 0
	for _, nd := range ix.nodes {
		if !nd.deleted {
			n++
		}
	}
	return n
}

// Dim returns the index dimensionality.
func (ix *Index) Dim() int { return ix.dim }

// Metric returns the configured distance metric.
func (ix *Index) Metric() vmath.Metric { return ix.metric }

// randomLevel draws a level from an exponentially decaying distribution, the
// mechanism that makes upper layers exponentially sparse.
func (ix *Index) randomLevel() int {
	r := ix.rng.Float64()
	return int(-math.Log(r) * ix.levelMult)
}

func (ix *Index) dist(a, b []float32) float32 {
	return vmath.Distance(ix.metric, a, b)
}

// Add inserts or replaces the vector for id. The slice is copied, so callers
// may reuse their buffer afterwards.
func (ix *Index) Add(id uint32, vec []float32) error {
	if len(vec) == 0 {
		return ErrEmptyVector
	}
	if len(vec) != ix.dim {
		return ErrDimMismatch
	}
	cp := make([]float32, len(vec))
	copy(cp, vec)

	ix.mu.Lock()
	defer ix.mu.Unlock()

	// Replace existing: drop it first so the graph is rebuilt cleanly.
	if _, ok := ix.nodes[id]; ok {
		ix.removeLocked(id)
	}

	level := ix.randomLevel()
	nd := &node{
		id:    id,
		vec:   cp,
		links: make([][]uint32, level+1),
	}
	ix.nodes[id] = nd

	// First node becomes the entry point.
	if !ix.hasEntry {
		ix.entryPoint = id
		ix.maxLevel = level
		ix.hasEntry = true
		return nil
	}

	ep := ix.entryPoint
	curDist := ix.dist(cp, ix.nodes[ep].vec)

	// Phase 1: from the top down to the node's own level + 1, greedily hop to
	// the closest neighbour on each layer to find a good entry point.
	for l := ix.maxLevel; l > level; l-- {
		ep, curDist = ix.greedyDescend(cp, ep, curDist, l)
	}

	// Phase 2: from the node's level down to 0, run a full ef-search and wire
	// up to the M best neighbours on each layer.
	for l := min(level, ix.maxLevel); l >= 0; l-- {
		candidates := ix.searchLayer(cp, ep, ix.efConstr, l)
		neighbours := ix.selectNeighbours(cp, candidates, ix.mForLayer(l))
		nd.links[l] = neighbours
		// Add the back-links and prune the neighbour if it now has too many.
		for _, nb := range neighbours {
			ix.connect(nb, id, l)
		}
		if len(candidates) > 0 {
			ep = candidates[0].id
		}
	}

	if level > ix.maxLevel {
		ix.maxLevel = level
		ix.entryPoint = id
	}
	return nil
}

// mForLayer returns the neighbour budget for a layer (2M on layer 0, M above).
func (ix *Index) mForLayer(l int) int {
	if l == 0 {
		return ix.mMax0
	}
	return ix.m
}

// greedyDescend walks to the closest node reachable on layer l, starting from
// ep. Returns the best node and its distance.
func (ix *Index) greedyDescend(q []float32, ep uint32, epDist float32, l int) (uint32, float32) {
	best := ep
	bestDist := epDist
	for {
		cur := ix.nodes[best]
		if l >= len(cur.links) {
			return best, bestDist
		}
		improved := false
		for _, nb := range cur.links[l] {
			nd := ix.nodes[nb]
			if nd == nil {
				continue
			}
			d := ix.dist(q, nd.vec)
			if d < bestDist {
				bestDist = d
				best = nb
				improved = true
			}
		}
		if !improved {
			return best, bestDist
		}
	}
}

// searchLayer runs the core HNSW best-first search on a single layer and
// returns up to ef candidates sorted nearest-first.
func (ix *Index) searchLayer(q []float32, entry uint32, ef, l int) []candidate {
	visited := make(map[uint32]struct{}, ef*2)
	entryNode := ix.nodes[entry]
	if entryNode == nil {
		return nil
	}
	d0 := ix.dist(q, entryNode.vec)

	frontier := &minHeap{{id: entry, dist: d0}} // nearest to expand next
	results := &maxHeap{{id: entry, dist: d0}}   // current best (worst on top)
	heap.Init(frontier)
	heap.Init(results)
	visited[entry] = struct{}{}

	for frontier.Len() > 0 {
		cur := heap.Pop(frontier).(candidate)
		// If the closest unexpanded candidate is farther than our current
		// worst result, no remaining candidate can improve the set.
		if results.Len() >= ef && cur.dist > results.peek().dist {
			break
		}
		curNode := ix.nodes[cur.id]
		if curNode == nil || l >= len(curNode.links) {
			continue
		}
		for _, nb := range curNode.links[l] {
			if _, seen := visited[nb]; seen {
				continue
			}
			visited[nb] = struct{}{}
			nd := ix.nodes[nb]
			if nd == nil {
				continue
			}
			d := ix.dist(q, nd.vec)
			if results.Len() < ef || d < results.peek().dist {
				heap.Push(frontier, candidate{id: nb, dist: d})
				heap.Push(results, candidate{id: nb, dist: d})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	// Drain results into ascending order (nearest first).
	out := make([]candidate, results.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(results).(candidate)
	}
	return out
}

// selectNeighbours picks the m closest candidates. This is the simple
// heuristic; it keeps the nearest neighbours, which is adequate for the
// metrics supported here.
func (ix *Index) selectNeighbours(_ []float32, candidates []candidate, m int) []uint32 {
	if len(candidates) > m {
		candidates = candidates[:m]
	}
	out := make([]uint32, 0, len(candidates))
	for _, c := range candidates {
		out = append(out, c.id)
	}
	return out
}

// connect adds dst to src's adjacency on layer l, pruning src back to its
// neighbour budget by keeping the closest links.
func (ix *Index) connect(src, dst uint32, l int) {
	sn := ix.nodes[src]
	if sn == nil || l >= len(sn.links) {
		return
	}
	// Avoid duplicates.
	for _, e := range sn.links[l] {
		if e == dst {
			return
		}
	}
	sn.links[l] = append(sn.links[l], dst)

	budget := ix.mForLayer(l)
	if len(sn.links[l]) <= budget {
		return
	}
	// Prune: keep the closest `budget` neighbours to src.
	cands := make([]candidate, 0, len(sn.links[l]))
	for _, nb := range sn.links[l] {
		nd := ix.nodes[nb]
		if nd == nil {
			continue
		}
		cands = append(cands, candidate{id: nb, dist: ix.dist(sn.vec, nd.vec)})
	}
	// Partial selection sort for the closest budget entries.
	for i := 0; i < budget && i < len(cands); i++ {
		minIdx := i
		for j := i + 1; j < len(cands); j++ {
			if cands[j].dist < cands[minIdx].dist {
				minIdx = j
			}
		}
		cands[i], cands[minIdx] = cands[minIdx], cands[i]
	}
	if len(cands) > budget {
		cands = cands[:budget]
	}
	pruned := make([]uint32, len(cands))
	for i, c := range cands {
		pruned[i] = c.id
	}
	sn.links[l] = pruned
}

// Result is a single search hit.
type Result struct {
	ID       uint32
	Distance float32
}

// Search returns up to k approximate nearest neighbours of q, nearest first.
func (ix *Index) Search(q []float32, k int) ([]Result, error) {
	if len(q) != ix.dim {
		return nil, ErrDimMismatch
	}
	if k <= 0 {
		return nil, nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	if !ix.hasEntry {
		return nil, nil
	}
	ep := ix.entryPoint
	curDist := ix.dist(q, ix.nodes[ep].vec)
	for l := ix.maxLevel; l > 0; l-- {
		ep, curDist = ix.greedyDescend(q, ep, curDist, l)
	}
	ef := ix.efSearch
	if ef < k {
		ef = k
	}
	candidates := ix.searchLayer(q, ep, ef, 0)

	out := make([]Result, 0, k)
	for _, c := range candidates {
		nd := ix.nodes[c.id]
		if nd == nil || nd.deleted {
			continue
		}
		out = append(out, Result{ID: c.id, Distance: c.dist})
		if len(out) == k {
			break
		}
	}
	return out, nil
}

// Delete marks an id as removed. The node stays in the graph for connectivity
// but is filtered from results; a later rebuild/compaction can drop it.
func (ix *Index) Delete(id uint32) error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	nd, ok := ix.nodes[id]
	if !ok {
		return ErrNotFound
	}
	nd.deleted = true
	return nil
}

// removeLocked fully detaches an id from the graph. Caller holds the write
// lock. Used when replacing an existing id on Add.
func (ix *Index) removeLocked(id uint32) {
	nd, ok := ix.nodes[id]
	if !ok {
		return
	}
	for l := range nd.links {
		for _, nb := range nd.links[l] {
			other := ix.nodes[nb]
			if other == nil || l >= len(other.links) {
				continue
			}
			filtered := other.links[l][:0]
			for _, e := range other.links[l] {
				if e != id {
					filtered = append(filtered, e)
				}
			}
			other.links[l] = filtered
		}
	}
	delete(ix.nodes, id)
	// If we removed the entry point, pick any remaining node.
	if ix.entryPoint == id {
		ix.hasEntry = false
		ix.maxLevel = 0
		for rid, rn := range ix.nodes {
			ix.entryPoint = rid
			ix.maxLevel = len(rn.links) - 1
			ix.hasEntry = true
			break
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
