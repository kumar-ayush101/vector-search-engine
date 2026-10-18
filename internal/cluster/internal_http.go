package cluster

import (
	"encoding/json"
	"net/http"

	"github.com/kumar-ayush101/vector-search-engine/internal/store"
)

// RegisterInternal mounts the internal shard endpoints on mux. Every node
// serves these so peer coordinators can reach its local shard. They are the
// server side of RemoteShard. Keep them off the public API surface (the main
// router mounts them under /internal, which is not exposed publicly in a real
// deployment — here they share the port for simplicity).
func RegisterInternal(mux *http.ServeMux, local *LocalShard) {
	coll := local.Collection()

	mux.HandleFunc("POST /internal/shard/upsert", func(w http.ResponseWriter, r *http.Request) {
		var it store.Item
		if err := json.NewDecoder(r.Body).Decode(&it); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := coll.Upsert(it); err != nil {
			writeShardErr(w, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("POST /internal/shard/query", func(w http.ResponseWriter, r *http.Request) {
		var req remoteQueryReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		hits, err := coll.Query(req.Vector, req.K)
		if err != nil {
			writeShardErr(w, err)
			return
		}
		writeJSON(w, remoteQueryResp{Hits: hits})
	})

	mux.HandleFunc("GET /internal/shard/item/{id}", func(w http.ResponseWriter, r *http.Request) {
		it, err := coll.Get(r.PathValue("id"))
		if err != nil {
			writeShardErr(w, err)
			return
		}
		writeJSON(w, it)
	})

	mux.HandleFunc("DELETE /internal/shard/item/{id}", func(w http.ResponseWriter, r *http.Request) {
		if err := coll.Delete(r.PathValue("id")); err != nil {
			writeShardErr(w, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("GET /internal/shard/count", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, remoteCountResp{Count: coll.Count()})
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func writeShardErr(w http.ResponseWriter, err error) {
	switch err {
	case store.ErrNotFound:
		http.Error(w, err.Error(), http.StatusNotFound)
	case store.ErrDimMismatch, store.ErrEmptyID:
		http.Error(w, err.Error(), http.StatusBadRequest)
	default:
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
