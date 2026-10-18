package httpapi

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kumar-ayush101/vector-search-engine/internal/embed"
	"github.com/kumar-ayush101/vector-search-engine/internal/hnsw"
	"github.com/kumar-ayush101/vector-search-engine/internal/store"
	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

const testDim = 32

func newTestServer(t *testing.T) http.Handler {
	t.Helper()
	params := hnsw.Params{Dim: testDim, Metric: vmath.Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1}
	s, err := store.New(params, "")
	if err != nil {
		t.Fatalf("store.New: %v", err)
	}
	emb := embed.NewHashEmbedder(testDim)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	h := NewHandlers(NewStoreEngine(s), testDim, emb, nil, log)
	return NewRouter(RouterConfig{Handlers: h, Log: log})
}

func do(t *testing.T, srv http.Handler, method, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatalf("encode: %v", err)
		}
	}
	req := httptest.NewRequest(method, path, &buf)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	return rec
}

func TestHealthReady(t *testing.T) {
	srv := newTestServer(t)
	for _, p := range []string{"/health", "/ready"} {
		rec := do(t, srv, http.MethodGet, p, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s = %d, want 200", p, rec.Code)
		}
	}
}

func TestUpsertAndQueryByText(t *testing.T) {
	srv := newTestServer(t)

	docs := map[string]string{
		"d1": "machine learning and neural networks",
		"d2": "cooking pasta with tomato sauce",
		"d3": "deep learning models and training",
	}
	for id, text := range docs {
		rec := do(t, srv, http.MethodPost, "/api/vectors", upsertRequest{ID: id, Text: text})
		if rec.Code != http.StatusOK {
			t.Fatalf("upsert %s = %d, body=%s", id, rec.Code, rec.Body.String())
		}
	}

	// Query with ML-related text; d1/d3 should outrank the cooking doc.
	rec := do(t, srv, http.MethodPost, "/api/query", queryRequest{Text: "neural network training", K: 3})
	if rec.Code != http.StatusOK {
		t.Fatalf("query = %d, body=%s", rec.Code, rec.Body.String())
	}
	var resp queryResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode query response: %v", err)
	}
	if len(resp.Hits) == 0 {
		t.Fatal("expected query hits")
	}
	if resp.Hits[0].ID == "d2" {
		t.Fatalf("cooking doc ranked first for an ML query: %+v", resp.Hits)
	}
}

func TestUpsertByVectorThenGetDelete(t *testing.T) {
	srv := newTestServer(t)
	vec := make([]float32, testDim)
	vec[0] = 1
	rec := do(t, srv, http.MethodPost, "/api/vectors", upsertRequest{
		ID: "x", Vector: vec, Metadata: map[string]string{"k": "v"},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("upsert = %d, body=%s", rec.Code, rec.Body.String())
	}

	rec = do(t, srv, http.MethodGet, "/api/vectors/x", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("get = %d", rec.Code)
	}
	var it store.Item
	json.Unmarshal(rec.Body.Bytes(), &it)
	if it.Metadata["k"] != "v" {
		t.Fatalf("metadata lost: %+v", it)
	}

	rec = do(t, srv, http.MethodDelete, "/api/vectors/x", nil)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("delete = %d, want 204", rec.Code)
	}
	rec = do(t, srv, http.MethodGet, "/api/vectors/x", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("get after delete = %d, want 404", rec.Code)
	}
}

func TestUpsertValidationErrors(t *testing.T) {
	srv := newTestServer(t)
	// Missing id.
	rec := do(t, srv, http.MethodPost, "/api/vectors", upsertRequest{Text: "hi"})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing id = %d, want 400", rec.Code)
	}
	// Neither vector nor text.
	rec = do(t, srv, http.MethodPost, "/api/vectors", upsertRequest{ID: "x"})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("no vector/text = %d, want 400", rec.Code)
	}
	// Both vector and text.
	rec = do(t, srv, http.MethodPost, "/api/vectors", upsertRequest{ID: "x", Text: "hi", Vector: []float32{1}})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("both vector/text = %d, want 400", rec.Code)
	}
}

func TestStats(t *testing.T) {
	srv := newTestServer(t)
	do(t, srv, http.MethodPost, "/api/vectors", upsertRequest{ID: "a", Text: "hello"})
	rec := do(t, srv, http.MethodGet, "/api/stats", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("stats = %d", rec.Code)
	}
	var st statsResponse
	json.Unmarshal(rec.Body.Bytes(), &st)
	if st.Count != 1 || st.Dim != testDim || st.Embedder != "hash-local" {
		t.Fatalf("stats = %+v", st)
	}
}
