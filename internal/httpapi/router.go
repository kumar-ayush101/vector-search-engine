package httpapi

import (
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/kumar-ayush101/vector-search-engine/internal/metrics"
)

// RouterConfig configures route registration.
type RouterConfig struct {
	Handlers    *Handlers
	Metrics     *metrics.Collectors
	Log         *slog.Logger
	AllowOrigin string

	// ServeFrontend serves a static SPA from FrontendDir at "/".
	ServeFrontend bool
	FrontendDir   string

	// Mount lets callers register extra routes (e.g. the cluster internal
	// shard endpoints) on the same mux before middleware is applied.
	Mount func(mux *http.ServeMux)
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

	// Extra routes (cluster internal endpoints).
	if cfg.Mount != nil {
		cfg.Mount(mux)
	}

	// Static frontend with SPA fallback (served last so API routes win).
	if cfg.ServeFrontend && cfg.FrontendDir != "" {
		mux.Handle("/", spaHandler(cfg.FrontendDir))
	}

	return chain(mux,
		recoverMiddleware(cfg.Log),
		requestIDMiddleware,
		corsMiddleware(cfg.AllowOrigin),
		observabilityMiddleware(cfg.Log, cfg.Metrics),
	)
}

// spaHandler serves static files from dir and falls back to index.html for
// client-side routes, so a single-page app's deep links work on refresh.
func spaHandler(dir string) http.Handler {
	fs := http.FileServer(http.Dir(dir))
	index := filepath.Join(dir, "index.html")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		clean := strings.TrimPrefix(filepath.Clean(r.URL.Path), string(os.PathSeparator))
		full := filepath.Join(dir, clean)
		if info, err := os.Stat(full); err == nil && !info.IsDir() {
			fs.ServeHTTP(w, r)
			return
		}
		if _, err := os.Stat(index); err != nil {
			writeError(w, http.StatusNotFound, "frontend not built")
			return
		}
		http.ServeFile(w, r, index)
	})
}
