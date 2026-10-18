package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http"
	"time"

	"github.com/kumar-ayush101/vector-search-engine/internal/metrics"
)

type ctxKey int

const requestIDKey ctxKey = 0

// statusRecorder captures the response status code for logging and metrics.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// middleware is a function that wraps an http.Handler.
type middleware func(http.Handler) http.Handler

// chain applies middleware in order so the first listed runs outermost.
func chain(h http.Handler, mws ...middleware) http.Handler {
	for i := len(mws) - 1; i >= 0; i-- {
		h = mws[i](h)
	}
	return h
}

// newRequestID returns a short random hex id.
func newRequestID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// requestIDMiddleware attaches a request id to the context and response header.
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = newRequestID()
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), requestIDKey, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// recoverMiddleware turns a panic into a 500 instead of crashing the server.
func recoverMiddleware(log *slog.Logger) middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					log.Error("panic recovered", "error", rec, "path", r.URL.Path)
					writeError(w, http.StatusInternalServerError, "internal server error")
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// corsMiddleware adds permissive CORS headers so a browser dashboard on another
// origin can call the API. Tighten the allowed origin for production use.
func corsMiddleware(origin string) middleware {
	if origin == "" {
		origin = "*"
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// observabilityMiddleware logs each request and records HTTP metrics.
func observabilityMiddleware(log *slog.Logger, m *metrics.Collectors) middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
			next.ServeHTTP(rec, r)
			dur := time.Since(start)

			route := r.Pattern
			if route == "" {
				route = r.URL.Path
			}
			if m != nil {
				m.HTTPRequests.WithLabelValues(r.Method, route, http.StatusText(rec.status)).Inc()
				m.HTTPDuration.WithLabelValues(r.Method, route).Observe(dur.Seconds())
			}
			log.Info("request",
				"method", r.Method, "path", r.URL.Path, "status", rec.status,
				"duration_ms", dur.Milliseconds(),
				"request_id", r.Context().Value(requestIDKey),
			)
		})
	}
}
