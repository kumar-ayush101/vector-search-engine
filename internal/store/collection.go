// Package store is the service layer that wraps the HNSW index with
// application concerns: user-facing string IDs, optional metadata, snapshot
// persistence and concurrency-safe access. The HTTP layer talks only to this
// package, never to the index directly.
package store

import (
	"errors"
	"sync"

	"github.com/kumar-ayush101/vector-search-engine/internal/hnsw"
	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

// Errors returned by the store.
var (
	ErrNotFound    = errors.New("store: id not found")
	ErrDimMismatch = errors.New("store: vector dimension mismatch")
	ErrEmptyID     = errors.New("store: empty id")
)

// Item is a stored vector with its user id and metadata.
type Item struct {
	ID       string            `json:"id"`
	Vector   []float32         `json:"vector,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// SearchHit is a single query result.
type SearchHit struct {
	ID       string            `json:"id"`
	Score    float32           `json:"score"`
	Distance float32           `json:"distance"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Collection is a named set of vectors backed by one HNSW index. The HNSW
// index keys on uint32, so the store maintains a bijection between user string
// ids and internal numeric ids.
type Collection struct {
	mu sync.RWMutex

	name   string
	index  *hnsw.Index
	metric vmath.Metric
	dim    int

	nextID   uint32
	byString map[string]uint32
	byUint   map[uint32]*Item
}

// NewCollection builds a collection with a fresh index.
func NewCollection(name string, p hnsw.Params) (*Collection, error) {
	ix, err := hnsw.NewIndex(p)
	if err != nil {
		return nil, err
	}
	return &Collection{
		name:     name,
		index:    ix,
		metric:   p.Metric,
		dim:      p.Dim,
		byString: make(map[string]uint32),
		byUint:   make(map[uint32]*Item),
	}, nil
}

// Name returns the collection name.
func (c *Collection) Name() string { return c.name }

// Dim returns the vector dimensionality.
func (c *Collection) Dim() int { return c.dim }

// Count returns the number of live items.
func (c *Collection) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.byString)
}

// Upsert inserts or replaces an item. The vector is copied by the index.
func (c *Collection) Upsert(it Item) error {
	if it.ID == "" {
		return ErrEmptyID
	}
	if len(it.Vector) != c.dim {
		return ErrDimMismatch
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	uid, ok := c.byString[it.ID]
	if !ok {
		uid = c.nextID
		c.nextID++
		c.byString[it.ID] = uid
	}
	if err := c.index.Add(uid, it.Vector); err != nil {
		if !ok {
			delete(c.byString, it.ID)
			c.nextID--
		}
		return err
	}
	// Store a copy of the item (without the raw vector to save memory; the
	// index owns the vector).
	stored := Item{ID: it.ID, Metadata: cloneMeta(it.Metadata)}
	c.byUint[uid] = &stored
	return nil
}

// Query returns the k nearest neighbours of the given vector. score is a
// convenience similarity (1 - distance for cosine; -distance otherwise) where
// larger is more similar.
func (c *Collection) Query(vec []float32, k int) ([]SearchHit, error) {
	if len(vec) != c.dim {
		return nil, ErrDimMismatch
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	raw, err := c.index.Search(vec, k)
	if err != nil {
		return nil, err
	}
	hits := make([]SearchHit, 0, len(raw))
	for _, r := range raw {
		it := c.byUint[r.ID]
		if it == nil {
			continue
		}
		hits = append(hits, SearchHit{
			ID:       it.ID,
			Distance: r.Distance,
			Score:    scoreFromDistance(c.metric, r.Distance),
			Metadata: it.Metadata,
		})
	}
	return hits, nil
}

// Get returns an item's metadata by id.
func (c *Collection) Get(id string) (Item, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	uid, ok := c.byString[id]
	if !ok {
		return Item{}, ErrNotFound
	}
	it := c.byUint[uid]
	return Item{ID: it.ID, Metadata: cloneMeta(it.Metadata)}, nil
}

// Delete removes an item by id.
func (c *Collection) Delete(id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	uid, ok := c.byString[id]
	if !ok {
		return ErrNotFound
	}
	_ = c.index.Delete(uid)
	delete(c.byString, id)
	delete(c.byUint, uid)
	return nil
}

func scoreFromDistance(m vmath.Metric, d float32) float32 {
	switch m {
	case vmath.Cosine:
		return 1 - d
	default:
		return -d
	}
}

func cloneMeta(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
