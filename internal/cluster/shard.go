package cluster

import (
	"context"

	"github.com/kumar-ayush101/vector-search-engine/internal/hnsw"
	"github.com/kumar-ayush101/vector-search-engine/internal/store"
)

// Shard is one partition of the vector space. Implementations may be local
// (in-process) or remote (another node over HTTP); the coordinator treats them
// uniformly.
type Shard interface {
	// Upsert stores an item on this shard.
	Upsert(ctx context.Context, it store.Item) error
	// Query returns this shard's local top-k for the vector.
	Query(ctx context.Context, vec []float32, k int) ([]store.SearchHit, error)
	// Get returns an item's metadata by id.
	Get(ctx context.Context, id string) (store.Item, error)
	// Delete removes an item by id.
	Delete(ctx context.Context, id string) error
	// Count returns the number of live items on this shard.
	Count(ctx context.Context) (int, error)
	// ID identifies the shard for logs and diagnostics.
	ID() string
}

// LocalShard is an in-process shard backed by a store.Collection. It is used
// both as a standalone shard and as the per-node storage behind a remote
// shard's HTTP handler.
type LocalShard struct {
	id   string
	coll *store.Collection
}

// NewLocalShard builds an in-process shard with its own HNSW index.
func NewLocalShard(id string, p hnsw.Params) (*LocalShard, error) {
	coll, err := store.NewCollection(id, p)
	if err != nil {
		return nil, err
	}
	return &LocalShard{id: id, coll: coll}, nil
}

// Collection exposes the backing collection (used to serve remote shard HTTP).
func (s *LocalShard) Collection() *store.Collection { return s.coll }

// ID implements Shard.
func (s *LocalShard) ID() string { return s.id }

// Upsert implements Shard.
func (s *LocalShard) Upsert(_ context.Context, it store.Item) error {
	return s.coll.Upsert(it)
}

// Query implements Shard.
func (s *LocalShard) Query(_ context.Context, vec []float32, k int) ([]store.SearchHit, error) {
	return s.coll.Query(vec, k)
}

// Get implements Shard.
func (s *LocalShard) Get(_ context.Context, id string) (store.Item, error) {
	return s.coll.Get(id)
}

// Delete implements Shard.
func (s *LocalShard) Delete(_ context.Context, id string) error {
	return s.coll.Delete(id)
}

// Count implements Shard.
func (s *LocalShard) Count(_ context.Context) (int, error) {
	return s.coll.Count(), nil
}

var _ Shard = (*LocalShard)(nil)
