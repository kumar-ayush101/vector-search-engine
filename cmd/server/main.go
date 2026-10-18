// Command server is the entrypoint for the vector search engine. It wires
// configuration, logging, the vector store, the embedding provider and the
// HTTP API. In single-node mode it serves one local index; in cluster mode
// (PEERS set) it builds a coordinator that shards across all peers and serves
// its own partition's internal shard endpoints for the other nodes.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/kumar-ayush101/vector-search-engine/internal/cluster"
	"github.com/kumar-ayush101/vector-search-engine/internal/config"
	"github.com/kumar-ayush101/vector-search-engine/internal/embed"
	"github.com/kumar-ayush101/vector-search-engine/internal/hnsw"
	"github.com/kumar-ayush101/vector-search-engine/internal/httpapi"
	"github.com/kumar-ayush101/vector-search-engine/internal/metrics"
	"github.com/kumar-ayush101/vector-search-engine/internal/store"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	log := newLogger(cfg)
	log.Info("starting vector search engine",
		"env", cfg.Env, "port", cfg.HTTPPort, "dim", cfg.Dim, "metric", cfg.Metric.String(),
		"cluster", cfg.ClusterEnabled(), "serve_frontend", cfg.ServeFrontend)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	params := hnsw.Params{
		Dim: cfg.Dim, Metric: cfg.Metric, M: cfg.M,
		EfConstruction: cfg.EfConstruction, EfSearch: cfg.EfSearch, Seed: cfg.Seed,
	}
	embedder := embed.NewHashEmbedder(cfg.Dim)
	m := metrics.New()

	var (
		engine     httpapi.Engine
		mount      func(mux *http.ServeMux)
		st         *store.Store // non-nil only in single-node mode (for snapshots)
	)

	if cfg.ClusterEnabled() {
		engine, mount, err = buildCluster(cfg, params, log)
		if err != nil {
			return err
		}
		log.Info("cluster mode", "shards", len(cfg.Peers), "node_index", cfg.NodeIndex)
	} else {
		st, err = store.New(params, cfg.SnapshotPath())
		if err != nil {
			return err
		}
		if restored, rerr := st.Restore(); rerr != nil {
			log.Warn("snapshot restore failed, starting empty", "error", rerr)
		} else if restored {
			log.Info("restored index from snapshot", "path", cfg.SnapshotPath(), "count", st.Default().Count())
		}
		engine = httpapi.NewStoreEngine(st)
		m.IndexSize.Set(float64(st.Default().Count()))
	}

	handlers := httpapi.NewHandlers(engine, cfg.Dim, embedder, m, log)
	router := httpapi.NewRouter(httpapi.RouterConfig{
		Handlers:      handlers,
		Metrics:       m,
		Log:           log,
		ServeFrontend: cfg.ServeFrontend,
		FrontendDir:   cfg.FrontendDir,
		Mount:         mount,
	})

	// Background periodic snapshots (single-node mode only).
	if st != nil && cfg.SnapshotEvery > 0 {
		go snapshotLoop(ctx, st, cfg.SnapshotEvery, log)
	}

	srv := &http.Server{
		Addr:              ":" + cfg.HTTPPort,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	serverErr := make(chan error, 1)
	go func() {
		log.Info("http server listening", "addr", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
		}
	}()

	select {
	case err := <-serverErr:
		return err
	case <-ctx.Done():
		log.Info("shutdown signal received")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown failed", "error", err)
	}
	if st != nil {
		if err := st.Snapshot(); err != nil {
			log.Error("final snapshot failed", "error", err)
		} else {
			log.Info("final snapshot written", "path", cfg.SnapshotPath())
		}
	}
	log.Info("shutdown complete")
	return nil
}

// buildCluster constructs a coordinator that shards across all peers. This
// node owns one local shard (at NodeIndex) and reaches the others as remote
// shards over HTTP; it also returns a mount function that serves this node's
// local shard to peers via the internal endpoints.
func buildCluster(cfg *config.Config, params hnsw.Params, log *slog.Logger) (httpapi.Engine, func(mux *http.ServeMux), error) {
	local, err := cluster.NewLocalShard("shard-local", params)
	if err != nil {
		return nil, nil, err
	}
	shards := make([]cluster.Shard, len(cfg.Peers))
	for i, peer := range cfg.Peers {
		if i == cfg.NodeIndex {
			shards[i] = local
		} else {
			shards[i] = cluster.NewRemoteShard("shard-"+itoa(i), peer)
		}
	}
	ring := cluster.NewRing(len(cfg.Peers), 128)
	coord := cluster.NewCoordinator(ring, shards)
	mount := func(mux *http.ServeMux) { cluster.RegisterInternal(mux, local) }
	return coord, mount, nil
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

func snapshotLoop(ctx context.Context, st *store.Store, every time.Duration, log *slog.Logger) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := st.Snapshot(); err != nil {
				log.Warn("periodic snapshot failed", "error", err)
			}
		}
	}
}

func newLogger(cfg *config.Config) *slog.Logger {
	level := slog.LevelDebug
	var handler slog.Handler
	if cfg.IsProduction() {
		level = slog.LevelInfo
		handler = slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level})
	} else {
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: level})
	}
	return slog.New(handler).With("service", "vector-search-engine")
}
