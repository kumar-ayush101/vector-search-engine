package hnsw

import (
	"math/rand"
	"sort"
	"testing"

	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

func newTestIndex(t *testing.T, dim int, m vmath.Metric) *Index {
	t.Helper()
	ix, err := NewIndex(Params{Dim: dim, Metric: m, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	return ix
}

func TestAddAndLen(t *testing.T) {
	ix := newTestIndex(t, 3, vmath.Cosine)
	for i := 0; i < 10; i++ {
		if err := ix.Add(uint32(i), []float32{float32(i), 0, 0}); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	if ix.Len() != 10 {
		t.Fatalf("Len = %d, want 10", ix.Len())
	}
}

func TestDimMismatch(t *testing.T) {
	ix := newTestIndex(t, 4, vmath.Cosine)
	if err := ix.Add(1, []float32{1, 2, 3}); err != ErrDimMismatch {
		t.Fatalf("Add wrong dim err = %v, want ErrDimMismatch", err)
	}
	if _, err := ix.Search([]float32{1, 2, 3}, 1); err != ErrDimMismatch {
		t.Fatalf("Search wrong dim err = %v, want ErrDimMismatch", err)
	}
}

func TestSearchExactMatch(t *testing.T) {
	ix := newTestIndex(t, 2, vmath.Euclidean)
	vecs := [][]float32{{0, 0}, {10, 10}, {5, 5}, {1, 1}}
	for i, v := range vecs {
		if err := ix.Add(uint32(i), v); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	res, err := ix.Search([]float32{10, 10}, 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("nearest to (10,10) = %+v, want id 1", res)
	}
}

func TestDeleteFiltersResult(t *testing.T) {
	ix := newTestIndex(t, 2, vmath.Euclidean)
	ix.Add(1, []float32{0, 0})
	ix.Add(2, []float32{1, 1})
	if err := ix.Delete(1); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if ix.Len() != 1 {
		t.Fatalf("Len after delete = %d, want 1", ix.Len())
	}
	res, _ := ix.Search([]float32{0, 0}, 5)
	for _, r := range res {
		if r.ID == 1 {
			t.Fatal("deleted id 1 returned in search")
		}
	}
	if err := ix.Delete(999); err != ErrNotFound {
		t.Fatalf("Delete missing = %v, want ErrNotFound", err)
	}
}

func TestReplaceExisting(t *testing.T) {
	ix := newTestIndex(t, 2, vmath.Euclidean)
	ix.Add(1, []float32{0, 0})
	ix.Add(2, []float32{100, 100})
	// Move id 1 right next to id 2.
	ix.Add(1, []float32{100, 101})
	if ix.Len() != 2 {
		t.Fatalf("Len after replace = %d, want 2", ix.Len())
	}
	res, _ := ix.Search([]float32{100, 100}, 2)
	if len(res) < 1 {
		t.Fatal("expected results after replace")
	}
}

// bruteForce returns the true k nearest ids for the query.
func bruteForce(vecs map[uint32][]float32, q []float32, m vmath.Metric, k int) []uint32 {
	type pair struct {
		id uint32
		d  float32
	}
	all := make([]pair, 0, len(vecs))
	for id, v := range vecs {
		all = append(all, pair{id, vmath.Distance(m, q, v)})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].d < all[j].d })
	out := make([]uint32, 0, k)
	for i := 0; i < k && i < len(all); i++ {
		out = append(out, all[i].id)
	}
	return out
}

// TestRecall builds a random dataset and checks that HNSW recovers a high
// fraction of the true nearest neighbours that brute force finds.
func TestRecall(t *testing.T) {
	const (
		n   = 2000
		dim = 32
		k   = 10
	)
	rng := rand.New(rand.NewSource(7))
	ix, err := NewIndex(Params{Dim: dim, Metric: vmath.Euclidean, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 3})
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	vecs := make(map[uint32][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		vecs[uint32(i)] = v
		if err := ix.Add(uint32(i), v); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	const queries = 50
	var hit, total int
	for qi := 0; qi < queries; qi++ {
		q := make([]float32, dim)
		for d := range q {
			q[d] = rng.Float32()
		}
		truth := bruteForce(vecs, q, vmath.Euclidean, k)
		truthSet := make(map[uint32]struct{}, k)
		for _, id := range truth {
			truthSet[id] = struct{}{}
		}
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		for _, r := range got {
			if _, ok := truthSet[r.ID]; ok {
				hit++
			}
		}
		total += k
	}
	recall := float64(hit) / float64(total)
	t.Logf("recall@%d = %.3f over %d queries", k, recall, queries)
	if recall < 0.90 {
		t.Fatalf("recall@%d = %.3f, want >= 0.90", k, recall)
	}
}
