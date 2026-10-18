package cluster

import (
	"container/heap"
	"context"
	"sort"
	"sync"

	"github.com/kumar-ayush101/vector-search-engine/internal/store"
)

// Coordinator routes writes to the owning shard and fans reads out to all
// shards, merging their partial results into a single global top-k. This is
// the classic scatter-gather pattern used by distributed search engines: each
// shard searches its own partition in parallel, and the coordinator keeps only
// the k best hits overall.
type Coordinator struct {
	ring   *Ring
	shards []Shard
}

// NewCoordinator wires a ring to its shards. len(shards) must equal
// ring.Shards().
func NewCoordinator(ring *Ring, shards []Shard) *Coordinator {
	return &Coordinator{ring: ring, shards: shards}
}

// Shards returns the number of shards.
func (c *Coordinator) Shards() int { return len(c.shards) }

// Upsert routes an item to the shard that owns its id.
func (c *Coordinator) Upsert(ctx context.Context, it store.Item) error {
	owner := c.ring.Owner(it.ID)
	return c.shards[owner].Upsert(ctx, it)
}

// Get fetches an item from its owning shard.
func (c *Coordinator) Get(ctx context.Context, id string) (store.Item, error) {
	owner := c.ring.Owner(id)
	return c.shards[owner].Get(ctx, id)
}

// Delete removes an item from its owning shard.
func (c *Coordinator) Delete(ctx context.Context, id string) error {
	owner := c.ring.Owner(id)
	return c.shards[owner].Delete(ctx, id)
}

// Count sums live item counts across all shards.
func (c *Coordinator) Count(ctx context.Context) (int, error) {
	total := 0
	for _, s := range c.shards {
		n, err := s.Count(ctx)
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

// Query scatters the search to every shard in parallel and gathers the global
// top-k. Each shard returns its own top-k; merging those is correct because a
// global top-k hit must be a top-k hit on whichever shard holds it.
func (c *Coordinator) Query(ctx context.Context, vec []float32, k int) ([]store.SearchHit, error) {
	type shardResult struct {
		hits []store.SearchHit
		err  error
	}
	results := make([]shardResult, len(c.shards))
	var wg sync.WaitGroup
	for i, s := range c.shards {
		wg.Add(1)
		go func(i int, s Shard) {
			defer wg.Done()
			hits, err := s.Query(ctx, vec, k)
			results[i] = shardResult{hits: hits, err: err}
		}(i, s)
	}
	wg.Wait()

	// Merge: keep the k hits with the smallest distance across all shards.
	// A max-heap of size k lets us drop the worst in O(log k) as we scan.
	h := &hitHeap{}
	heap.Init(h)
	for _, r := range results {
		if r.err != nil {
			return nil, r.err
		}
		for _, hit := range r.hits {
			if h.Len() < k {
				heap.Push(h, hit)
			} else if hit.Distance < (*h)[0].Distance {
				heap.Pop(h)
				heap.Push(h, hit)
			}
		}
	}
	out := make([]store.SearchHit, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(store.SearchHit)
	}
	// out is now ascending by distance (nearest first).
	sort.SliceStable(out, func(i, j int) bool { return out[i].Distance < out[j].Distance })
	return out, nil
}

// hitHeap is a max-heap by Distance: the farthest hit is on top so it can be
// evicted when a closer one arrives.
type hitHeap []store.SearchHit

func (h hitHeap) Len() int            { return len(h) }
func (h hitHeap) Less(i, j int) bool  { return h[i].Distance > h[j].Distance }
func (h hitHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *hitHeap) Push(x any)         { *h = append(*h, x.(store.SearchHit)) }
func (h *hitHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}
