package cluster

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kumar-ayush101/vector-search-engine/internal/store"
)

// TestRemoteShardCluster builds a cluster whose shards are reached over real
// HTTP (httptest servers), proving the RemoteShard client and internal
// endpoints form a working distributed cluster — the same coordinator logic,
// now across processes.
func TestRemoteShardCluster(t *testing.T) {
	const (
		numShards = 3
		dim       = 16
		n         = 300
		k         = 5
	)
	ctx := context.Background()

	// Stand up numShards nodes, each serving its local shard over HTTP.
	servers := make([]*httptest.Server, numShards)
	shards := make([]Shard, numShards)
	for i := 0; i < numShards; i++ {
		local, err := NewLocalShard(fmt.Sprintf("shard-%d", i), params(dim))
		if err != nil {
			t.Fatalf("NewLocalShard: %v", err)
		}
		mux := http.NewServeMux()
		RegisterInternal(mux, local)
		srv := httptest.NewServer(mux)
		servers[i] = srv
		shards[i] = NewRemoteShard(fmt.Sprintf("remote-%d", i), srv.URL)
	}
	defer func() {
		for _, s := range servers {
			s.Close()
		}
	}()

	ring := NewRing(numShards, 128)
	coord := NewCoordinator(ring, shards)

	// Upsert through the coordinator -> routed to owning remote shard over HTTP.
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		v[i%dim] = float32(i)
		it := store.Item{ID: fmt.Sprintf("v-%d", i), Vector: v, Metadata: map[string]string{"i": fmt.Sprint(i)}}
		if err := coord.Upsert(ctx, it); err != nil {
			t.Fatalf("remote upsert %d: %v", i, err)
		}
	}

	// Count across all remote shards must equal n.
	total, err := coord.Count(ctx)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if total != n {
		t.Fatalf("remote cluster count = %d, want %d", total, n)
	}

	// Query flows out to all remote shards and merges.
	q := make([]float32, dim)
	q[0] = 1
	hits, err := coord.Query(ctx, q, k)
	if err != nil {
		t.Fatalf("remote query: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected hits from remote cluster")
	}

	// Get + Delete round-trip over HTTP.
	got, err := coord.Get(ctx, "v-1")
	if err != nil || got.Metadata["i"] != "1" {
		t.Fatalf("remote Get = %+v, %v", got, err)
	}
	if err := coord.Delete(ctx, "v-1"); err != nil {
		t.Fatalf("remote Delete: %v", err)
	}
	if _, err := coord.Get(ctx, "v-1"); err != store.ErrNotFound {
		t.Fatalf("remote Get after delete = %v, want ErrNotFound", err)
	}
}
