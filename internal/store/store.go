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

// LiveVectors returns the number of vectors in the underlying index graph,
// including soft-deleted tombstones not yet compacted. Count() returns the
// number of live user-visible items.
func (s *Store) LiveVectors() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.coll.index.Len()
}

// metaPath returns the sidecar metadata path for the index snapshot.
func (s *Store) metaPath() string {
	if s.snapshot == "" {
		return ""
	}
	return s.snapshot + ".meta.json"
}

// Snapshot persists the current index and its metadata sidecar to disk
// atomically. The index graph and the string-id/metadata map are written as
// two atomically-renamed files so a restart restores the collection fully.
func (s *Store) Snapshot() error {
	s.mu.RLock()
	coll := s.coll
	path := s.snapshot
	meta := s.metaPath()
	s.mu.RUnlock()
	if path == "" {
		return nil
	}
	if err := coll.index.SaveDir(path); err != nil {
		return err
	}
	return coll.saveMeta(meta)
}

// Restore loads an index snapshot and its metadata sidecar if present. It is a
// no-op (returns false, nil) when no snapshot file exists, so a fresh
// deployment starts cleanly. The vector graph and the string-id/metadata map
// are both restored, so queries and Get return correct ids and metadata after
// a restart.
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
	// Restore the metadata sidecar if it exists. A missing sidecar is
	// tolerated (older snapshots): the graph still loads, metadata is empty.
	if meta := s.metaPath(); meta != "" {
		if err := s.coll.loadMeta(meta); err != nil && !os.IsNotExist(err) {
			return false, err
		}
	}
	return true, nil
}
