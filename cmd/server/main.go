// Command server is the entrypoint for the vector search engine. It wires
// configuration, logging, the vector store, the embedding provider and the
// HTTP API, restores any on-disk snapshot, runs periodic background snapshots,
// and shuts down gracefully on SIGINT/SIGTERM.
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
		"env", cfg.Env, "port", cfg.HTTPPort, "dim", cfg.Dim, "metric", cfg.Metric.String())

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Vector store.
	params := hnsw.Params{
		Dim: cfg.Dim, Metric: cfg.Metric, M: cfg.M,
		EfConstruction: cfg.EfConstruction, EfSearch: cfg.EfSearch, Seed: cfg.Seed,
	}
	st, err := store.New(params, cfg.SnapshotPath())
	if err != nil {
		return err
	}
	if restored, err := st.Restore(); err != nil {
		log.Warn("snapshot restore failed, starting empty", "error", err)
	} else if restored {
		log.Info("restored index from snapshot", "path", cfg.SnapshotPath(), "count", st.LiveVectors())
	}

	// Embedding provider: deterministic local embedder by default.
	embedder := embed.NewHashEmbedder(cfg.Dim)

	// Metrics + handlers + router.
	m := metrics.New()
	m.IndexSize.Set(float64(st.Default().Count()))
	handlers := httpapi.NewHandlers(st, embedder, m, log)
	router := httpapi.NewRouter(httpapi.RouterConfig{Handlers: handlers, Metrics: m, Log: log})

	// Background periodic snapshot.
	if cfg.SnapshotEvery > 0 {
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

	// Graceful shutdown: stop accepting requests, then persist a final snapshot.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown failed", "error", err)
	}
	if err := st.Snapshot(); err != nil {
		log.Error("final snapshot failed", "error", err)
	} else {
		log.Info("final snapshot written", "path", cfg.SnapshotPath())
	}
	log.Info("shutdown complete")
	return nil
}

// snapshotLoop persists the index at a fixed interval until ctx is cancelled.
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
