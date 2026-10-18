package httpapi

import (
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/kumar-ayush101/vector-search-engine/internal/embed"
	"github.com/kumar-ayush101/vector-search-engine/internal/metrics"
	"github.com/kumar-ayush101/vector-search-engine/internal/store"
)

// Handlers holds the dependencies for the HTTP handlers.
type Handlers struct {
	store    *store.Store
	embedder embed.Provider
	metrics  *metrics.Collectors
	log      *slog.Logger
}

// NewHandlers constructs the handler set.
func NewHandlers(s *store.Store, e embed.Provider, m *metrics.Collectors, log *slog.Logger) *Handlers {
	return &Handlers{store: s, embedder: e, metrics: m, log: log}
}

// ---- Request/response bodies ----

type upsertRequest struct {
	ID       string            `json:"id"`
	Vector   []float32         `json:"vector"`
	Text     string            `json:"text"`
	Metadata map[string]string `json:"metadata"`
}

type queryRequest struct {
	Vector []float32 `json:"vector"`
	Text   string    `json:"text"`
	K      int       `json:"k"`
}

type queryResponse struct {
	Hits []store.SearchHit `json:"hits"`
	K    int               `json:"k"`
}

type statsResponse struct {
	Count    int    `json:"count"`
	Dim      int    `json:"dim"`
	Embedder string `json:"embedder"`
}

// ---- Handlers ----

// Upsert inserts or replaces a vector. The body may carry a raw "vector" or a
// "text" field that is embedded server-side via the configured provider.
func (h *Handlers) Upsert(w http.ResponseWriter, r *http.Request) {
	var req upsertRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.ID == "" {
		writeError(w, http.StatusBadRequest, "id is required")
		return
	}
	vec, err := h.resolveVector(r, req.Vector, req.Text)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := h.store.Default().Upsert(store.Item{ID: req.ID, Vector: vec, Metadata: req.Metadata}); err != nil {
		h.writeStoreError(w, err)
		return
	}
	if h.metrics != nil {
		h.metrics.UpsertsTotal.Inc()
		h.metrics.IndexSize.Set(float64(h.store.Default().Count()))
	}
	writeJSON(w, http.StatusOK, store.Item{ID: req.ID, Metadata: req.Metadata})
}

// Query returns the k nearest neighbours of a vector or embedded text.
func (h *Handlers) Query(w http.ResponseWriter, r *http.Request) {
	var req queryRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.K <= 0 {
		req.K = 10
	}
	vec, err := h.resolveVector(r, req.Vector, req.Text)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	start := time.Now()
	hits, err := h.store.Default().Query(vec, req.K)
	if err != nil {
		h.writeStoreError(w, err)
		return
	}
	if h.metrics != nil {
		h.metrics.QueriesTotal.Inc()
		h.metrics.QueryDuration.Observe(time.Since(start).Seconds())
	}
	writeJSON(w, http.StatusOK, queryResponse{Hits: hits, K: req.K})
}

// Get returns an item's metadata by id.
func (h *Handlers) Get(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	it, err := h.store.Default().Get(id)
	if err != nil {
		h.writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, it)
}

// Delete removes an item by id.
func (h *Handlers) Delete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := h.store.Default().Delete(id); err != nil {
		h.writeStoreError(w, err)
		return
	}
	if h.metrics != nil {
		h.metrics.IndexSize.Set(float64(h.store.Default().Count()))
	}
	w.WriteHeader(http.StatusNoContent)
}

// Stats returns collection statistics.
func (h *Handlers) Stats(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, statsResponse{
		Count:    h.store.Default().Count(),
		Dim:      h.store.Default().Dim(),
		Embedder: h.embedder.Name(),
	})
}

// Health is a liveness probe.
func (h *Handlers) Health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// Ready is a readiness probe; the store is always ready once constructed.
func (h *Handlers) Ready(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ready",
		"count":  h.store.Default().Count(),
	})
}

// resolveVector returns the vector to use: an explicit vector if present,
// otherwise the embedding of the text field. Exactly one must be provided.
func (h *Handlers) resolveVector(r *http.Request, vec []float32, text string) ([]float32, error) {
	switch {
	case len(vec) > 0 && text != "":
		return nil, errors.New("provide either 'vector' or 'text', not both")
	case len(vec) > 0:
		return vec, nil
	case text != "":
		return h.embedder.Embed(r.Context(), text)
	default:
		return nil, errors.New("provide a 'vector' or 'text' field")
	}
}

// writeStoreError maps store errors to HTTP status codes.
func (h *Handlers) writeStoreError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, store.ErrNotFound):
		writeError(w, http.StatusNotFound, err.Error())
	case errors.Is(err, store.ErrDimMismatch), errors.Is(err, store.ErrEmptyID):
		writeError(w, http.StatusBadRequest, err.Error())
	default:
		h.log.Error("store error", "error", err)
		writeError(w, http.StatusInternalServerError, "internal error")
	}
}
