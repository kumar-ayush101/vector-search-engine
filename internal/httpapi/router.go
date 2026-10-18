package httpapi

import (
	"log/slog"
	"net/http"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/kumar-ayush101/vector-search-engine/internal/metrics"
)

// RouterConfig configures route registration.
type RouterConfig struct {
	Handlers    *Handlers
	Metrics     *metrics.Collectors
	Log         *slog.Logger
	AllowOrigin string
}

// NewRouter builds the fully wired HTTP handler with middleware applied.
func NewRouter(cfg RouterConfig) http.Handler {
	mux := http.NewServeMux()
	h := cfg.Handlers

	// Health & readiness.
	mux.HandleFunc("GET /health", h.Health)
	mux.HandleFunc("GET /ready", h.Ready)

	// Observability.
	mux.Handle("GET /metrics", promhttp.Handler())

	// Vector API (Go 1.22+ method + path-value routing).
	mux.HandleFunc("POST /api/vectors", h.Upsert)
	mux.HandleFunc("GET /api/vectors/{id}", h.Get)
	mux.HandleFunc("DELETE /api/vectors/{id}", h.Delete)
	mux.HandleFunc("POST /api/query", h.Query)
	mux.HandleFunc("GET /api/stats", h.Stats)

	return chain(mux,
		recoverMiddleware(cfg.Log),
		requestIDMiddleware,
		corsMiddleware(cfg.AllowOrigin),
		observabilityMiddleware(cfg.Log, cfg.Metrics),
	)
}
