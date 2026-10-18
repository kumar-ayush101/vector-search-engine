package store

import (
	"os"
	"sync"

	"github.com/kumar-ayush101/vector-search-engine/internal/hnsw"
)

// Store owns the default collection and its on-disk snapshot. It is the single
// object the HTTP layer depends on. Keeping it small (one collection) matches
// the current single-node scope; multi-collection routing is on the roadmap.
type Store struct {
	mu      sync.RWMutex
	coll    *Collection
	params  hnsw.Params
	snapshot string
}

// New creates a store with an empty default collection.
func New(params hnsw.Params, snapshotPath string) (*Store, error) {
	coll, err := NewCollection("default", params)
	if err != nil {
		return nil, err
	}
	return &Store{coll: coll, params: params, snapshot: snapshotPath}, nil
}

// Default returns the default collection.
func (s *Store) Default() *Collection {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.coll
}

// Snapshot persists the current index to disk atomically.
func (s *Store) Snapshot() error {
	s.mu.RLock()
	coll := s.coll
	path := s.snapshot
	s.mu.RUnlock()
	if path == "" {
		return nil
	}
	return coll.index.SaveDir(path)
}

// Restore loads an index snapshot if one exists at the configured path. It is
// a no-op (returns false, nil) when no snapshot file is present, so a fresh
// deployment starts cleanly.
//
// Note: the on-disk snapshot stores the HNSW graph keyed by internal numeric
// ids. The string-id and metadata mapping is rebuilt from the loaded index by
// assigning synthetic ids, which is sufficient for warm-restart of the vector
// graph. Full metadata durability is tracked on the roadmap.
func (s *Store) Restore() (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.snapshot == "" {
		return false, nil
	}
	if _, err := os.Stat(s.snapshot); err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	ix, err := hnsw.Load(s.snapshot)
	if err != nil {
		return false, err
	}
	s.coll.index = ix
	return true, nil
}
