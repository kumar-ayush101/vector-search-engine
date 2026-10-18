package store

import (
	"path/filepath"
	"testing"

	"github.com/kumar-ayush101/vector-search-engine/internal/hnsw"
	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

func testParams(dim int) hnsw.Params {
	return hnsw.Params{Dim: dim, Metric: vmath.Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 5}
}

func TestUpsertQuery(t *testing.T) {
	c, err := NewCollection("t", testParams(3))
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	items := []Item{
		{ID: "a", Vector: []float32{1, 0, 0}, Metadata: map[string]string{"color": "red"}},
		{ID: "b", Vector: []float32{0, 1, 0}},
		{ID: "c", Vector: []float32{0, 0, 1}},
	}
	for _, it := range items {
		if err := c.Upsert(it); err != nil {
			t.Fatalf("Upsert %s: %v", it.ID, err)
		}
	}
	if c.Count() != 3 {
		t.Fatalf("Count = %d, want 3", c.Count())
	}
	hits, err := c.Query([]float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != "a" {
		t.Fatalf("nearest = %+v, want id a", hits)
	}
	if hits[0].Metadata["color"] != "red" {
		t.Fatalf("metadata not preserved: %+v", hits[0].Metadata)
	}
	// Cosine score of identical direction should be ~1.
	if hits[0].Score < 0.99 {
		t.Fatalf("score = %v, want ~1", hits[0].Score)
	}
}

func TestUpsertValidation(t *testing.T) {
	c, _ := NewCollection("t", testParams(3))
	if err := c.Upsert(Item{ID: "", Vector: []float32{1, 2, 3}}); err != ErrEmptyID {
		t.Fatalf("empty id err = %v, want ErrEmptyID", err)
	}
	if err := c.Upsert(Item{ID: "x", Vector: []float32{1, 2}}); err != ErrDimMismatch {
		t.Fatalf("dim mismatch err = %v, want ErrDimMismatch", err)
	}
}

func TestGetDelete(t *testing.T) {
	c, _ := NewCollection("t", testParams(2))
	c.Upsert(Item{ID: "x", Vector: []float32{1, 1}, Metadata: map[string]string{"k": "v"}})
	got, err := c.Get("x")
	if err != nil || got.Metadata["k"] != "v" {
		t.Fatalf("Get = %+v, %v", got, err)
	}
	if err := c.Delete("x"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := c.Get("x"); err != ErrNotFound {
		t.Fatalf("Get after delete = %v, want ErrNotFound", err)
	}
	if c.Count() != 0 {
		t.Fatalf("Count after delete = %d, want 0", c.Count())
	}
}

func TestUpsertReplace(t *testing.T) {
	c, _ := NewCollection("t", testParams(2))
	c.Upsert(Item{ID: "x", Vector: []float32{1, 0}})
	c.Upsert(Item{ID: "x", Vector: []float32{0, 1}, Metadata: map[string]string{"v": "2"}})
	if c.Count() != 1 {
		t.Fatalf("Count after replace = %d, want 1", c.Count())
	}
	got, _ := c.Get("x")
	if got.Metadata["v"] != "2" {
		t.Fatalf("metadata not updated on replace: %+v", got.Metadata)
	}
}

func TestStoreSnapshotRestore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.vsnap")
	s, err := New(testParams(4), path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 20; i++ {
		v := []float32{float32(i), 1, 2, 3}
		if err := s.Default().Upsert(Item{ID: string(rune('a' + i)), Vector: v}); err != nil {
			t.Fatalf("Upsert: %v", err)
		}
	}
	if err := s.Snapshot(); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// New store restores the index graph from disk.
	s2, _ := New(testParams(4), path)
	ok, err := s2.Restore()
	if err != nil || !ok {
		t.Fatalf("Restore = %v, %v; want true, nil", ok, err)
	}
	if s2.Default().index.Len() != 20 {
		t.Fatalf("restored index len = %d, want 20", s2.Default().index.Len())
	}
}

func TestRestoreNoSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.vsnap")
	s, _ := New(testParams(4), path)
	ok, err := s.Restore()
	if err != nil || ok {
		t.Fatalf("Restore with no file = %v, %v; want false, nil", ok, err)
	}
}
