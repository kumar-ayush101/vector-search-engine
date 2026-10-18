// Package httpapi exposes the vector search engine over a JSON REST API built
// on the net/http standard library (Go 1.22+ method+path routing), with no web
// framework.
package httpapi

import (
	"encoding/json"
	"net/http"
)

// errorResponse is the JSON body returned for any error.
type errorResponse struct {
	Error string `json:"error"`
}

// writeJSON serialises v as JSON with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if v != nil {
		_ = json.NewEncoder(w).Encode(v)
	}
}

// writeError writes a JSON error body with the given status code.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, errorResponse{Error: msg})
}

// decodeJSON reads and strictly decodes the request body into dst, rejecting
// unknown fields so malformed requests fail loudly instead of silently.
func decodeJSON(r *http.Request, dst any) error {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	return dec.Decode(dst)
}
