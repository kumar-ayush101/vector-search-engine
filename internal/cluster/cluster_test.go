package cluster

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/kumar-ayush101/vector-search-engine/internal/hnsw"
	"github.com/kumar-ayush101/vector-search-engine/internal/store"
	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

func params(dim int) hnsw.Params {
	return hnsw.Params{Dim: dim, Metric: vmath.Euclidean, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1}
}

func TestRingDistribution(t *testing.T) {
	ring := NewRing(4, 256)
	keys := make([]string, 10000)
	for i := range keys {
		keys[i] = fmt.Sprintf("key-%d", i)
	}
	dist := ring.Distribution(keys)
	if len(dist) != 4 {
		t.Fatalf("distribution len = %d, want 4", len(dist))
	}
	// Every shard should get a non-trivial share (within 2x of even).
	even := len(keys) / 4
	for s, c := range dist {
		if c < even/2 || c > even*2 {
			t.Fatalf("shard %d got %d keys, want roughly %d", s, c, even)
		}
	}
}

func TestRingDeterministic(t *testing.T) {
	ring := NewRing(3, 128)
	for i := 0; i < 100; i++ {
		k := fmt.Sprintf("item-%d", i)
		if ring.Owner(k) != ring.Owner(k) {
			t.Fatalf("owner not deterministic for %s", k)
		}
	}
}

func buildCoordinator(t *testing.T, numShards, dim int) *Coordinator {
	t.Helper()
	ring := NewRing(numShards, 128)
	shards := make([]Shard, numShards)
	for i := 0; i < numShards; i++ {
		s, err := NewLocalShard(fmt.Sprintf("shard-%d", i), params(dim))
		if err != nil {
			t.Fatalf("NewLocalShard: %v", err)
		}
		shards[i] = s
	}
	return NewCoordinator(ring, shards)
}

// TestScatterGatherMatchesSingleNode is the key correctness test: a sharded
// cluster must return the same nearest neighbours as a single-node index over
// the identical data, for the vast majority of queries.
func TestScatterGatherMatchesSingleNode(t *testing.T) {
	const (
		n   = 2000
		dim = 32
		k   = 10
	)
	rng := rand.New(rand.NewSource(42))
	ctx := context.Background()

	// Single-node baseline.
	single, err := store.NewCollection("baseline", params(dim))
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	// Sharded cluster over 4 shards.
	coord := buildCoordinator(t, 4, dim)

	data := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		data[i] = v
		id := fmt.Sprintf("v-%d", i)
		it := store.Item{ID: id, Vector: v}
		if err := single.Upsert(it); err != nil {
			t.Fatalf("baseline upsert: %v", err)
		}
		if err := coord.Upsert(ctx, it); err != nil {
			t.Fatalf("cluster upsert: %v", err)
		}
	}

	// Every shard should hold some data.
	total, _ := coord.Count(ctx)
	if total != n {
		t.Fatalf("cluster total count = %d, want %d", total, n)
	}

	// Compare top-k ids on many queries. Both use the same approximate index,
	// so allow a small mismatch margin from ANN non-determinism across shards.
	const queries = 200
	var agree, compared int
	for q := 0; q < queries; q++ {
		qv := make([]float32, dim)
		for d := range qv {
			qv[d] = rng.Float32()
		}
		base, err := single.Query(qv, k)
		if err != nil {
			t.Fatalf("baseline query: %v", err)
		}
		clu, err := coord.Query(ctx, qv, k)
		if err != nil {
			t.Fatalf("cluster query: %v", err)
		}
		baseSet := make(map[string]struct{}, len(base))
		for _, h := range base {
			baseSet[h.ID] = struct{}{}
		}
		for _, h := range clu {
			if _, ok := baseSet[h.ID]; ok {
				agree++
			}
		}
		compared += k
	}
	overlap := float64(agree) / float64(compared)
	t.Logf("cluster vs single-node top-%d agreement = %.3f", k, overlap)
	if overlap < 0.95 {
		t.Fatalf("cluster/single-node agreement = %.3f, want >= 0.95", overlap)
	}
}

func TestCoordinatorGetDelete(t *testing.T) {
	ctx := context.Background()
	coord := buildCoordinator(t, 3, 4)
	it := store.Item{ID: "abc", Vector: []float32{1, 2, 3, 4}, Metadata: map[string]string{"k": "v"}}
	if err := coord.Upsert(ctx, it); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	got, err := coord.Get(ctx, "abc")
	if err != nil || got.Metadata["k"] != "v" {
		t.Fatalf("Get = %+v, %v", got, err)
	}
	if err := coord.Delete(ctx, "abc"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := coord.Get(ctx, "abc"); err != store.ErrNotFound {
		t.Fatalf("Get after delete = %v, want ErrNotFound", err)
	}
}
