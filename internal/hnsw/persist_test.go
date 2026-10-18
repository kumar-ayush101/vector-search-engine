package hnsw

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

func writeBytes(path string, b []byte) error {
	return os.WriteFile(path, b, 0o644)
}

func TestSaveLoadRoundTrip(t *testing.T) {
	const (
		n   = 500
		dim = 16
		k   = 10
	)
	rng := rand.New(rand.NewSource(11))
	ix, err := NewIndex(Params{Dim: dim, Metric: vmath.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9})
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	queries := make([][]float32, 20)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		if err := ix.Add(uint32(i), v); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	for qi := range queries {
		q := make([]float32, dim)
		for d := range q {
			q[d] = rng.Float32()
		}
		queries[qi] = q
	}

	path := filepath.Join(t.TempDir(), "index.vsnap")
	if err := ix.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Len() != ix.Len() {
		t.Fatalf("loaded Len = %d, want %d", loaded.Len(), ix.Len())
	}
	if loaded.Dim() != dim || loaded.Metric() != vmath.Cosine {
		t.Fatalf("loaded params mismatch: dim=%d metric=%v", loaded.Dim(), loaded.Metric())
	}

	// The reloaded index must return identical search results.
	for _, q := range queries {
		a, _ := ix.Search(q, k)
		b, _ := loaded.Search(q, k)
		if len(a) != len(b) {
			t.Fatalf("result length differs: %d vs %d", len(a), len(b))
		}
		for i := range a {
			if a[i].ID != b[i].ID {
				t.Fatalf("result %d differs after reload: %d vs %d", i, a[i].ID, b[i].ID)
			}
		}
	}
}

func TestLoadBadMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.vsnap")
	if err := writeBytes(path, []byte{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("setup: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("Load with bad magic expected error")
	}
}
