package httpapi

import (
	"context"

	"github.com/kumar-ayush101/vector-search-engine/internal/store"
)

// Engine is the read/write surface the HTTP handlers depend on. It is
// satisfied by both the single-node store (via storeEngine) and the cluster
// coordinator, so the same handlers serve both deployment modes.
type Engine interface {
	Upsert(ctx context.Context, it store.Item) error
	Query(ctx context.Context, vec []float32, k int) ([]store.SearchHit, error)
	Get(ctx context.Context, id string) (store.Item, error)
	Delete(ctx context.Context, id string) error
	Count(ctx context.Context) (int, error)
}

// storeEngine adapts a single-node *store.Store to the Engine interface.
type storeEngine struct{ s *store.Store }

// NewStoreEngine wraps a store as an Engine for single-node mode.
func NewStoreEngine(s *store.Store) Engine { return &storeEngine{s: s} }

func (e *storeEngine) Upsert(_ context.Context, it store.Item) error {
	return e.s.Default().Upsert(it)
}

func (e *storeEngine) Query(_ context.Context, vec []float32, k int) ([]store.SearchHit, error) {
	return e.s.Default().Query(vec, k)
}

func (e *storeEngine) Get(_ context.Context, id string) (store.Item, error) {
	return e.s.Default().Get(id)
}

func (e *storeEngine) Delete(_ context.Context, id string) error {
	return e.s.Default().Delete(id)
}

func (e *storeEngine) Count(_ context.Context) (int, error) {
	return e.s.Default().Count(), nil
}
