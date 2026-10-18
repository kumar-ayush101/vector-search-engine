package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/kumar-ayush101/vector-search-engine/internal/store"
)

// RemoteShard is a Shard that lives on another node, reached over HTTP. The
// coordinator on the lead node holds one RemoteShard per peer; each peer runs
// the same binary and serves the internal shard endpoints (see InternalMux).
// This turns the in-process scatter-gather into a real multi-node cluster
// without changing the coordinator logic.
type RemoteShard struct {
	id      string
	baseURL string
	client  *http.Client
}

// NewRemoteShard builds a client for a peer node. baseURL is like
// "http://node2:8080".
func NewRemoteShard(id, baseURL string) *RemoteShard {
	return &RemoteShard{
		id:      id,
		baseURL: baseURL,
		client:  &http.Client{Timeout: 5 * time.Second},
	}
}

// ID implements Shard.
func (r *RemoteShard) ID() string { return r.id }

// Upsert implements Shard.
func (r *RemoteShard) Upsert(ctx context.Context, it store.Item) error {
	return r.post(ctx, "/internal/shard/upsert", it, nil)
}

type remoteQueryReq struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
}

type remoteQueryResp struct {
	Hits []store.SearchHit `json:"hits"`
}

// Query implements Shard.
func (r *RemoteShard) Query(ctx context.Context, vec []float32, k int) ([]store.SearchHit, error) {
	var resp remoteQueryResp
	if err := r.post(ctx, "/internal/shard/query", remoteQueryReq{Vector: vec, K: k}, &resp); err != nil {
		return nil, err
	}
	return resp.Hits, nil
}

// Get implements Shard.
func (r *RemoteShard) Get(ctx context.Context, id string) (store.Item, error) {
	var it store.Item
	err := r.get(ctx, "/internal/shard/item/"+id, &it)
	return it, err
}

// Delete implements Shard.
func (r *RemoteShard) Delete(ctx context.Context, id string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, r.baseURL+"/internal/shard/item/"+id, nil)
	if err != nil {
		return err
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return store.ErrNotFound
	}
	if resp.StatusCode >= 300 {
		return fmt.Errorf("remote shard %s delete: status %d", r.id, resp.StatusCode)
	}
	return nil
}

type remoteCountResp struct {
	Count int `json:"count"`
}

// Count implements Shard.
func (r *RemoteShard) Count(ctx context.Context) (int, error) {
	var resp remoteCountResp
	if err := r.get(ctx, "/internal/shard/count", &resp); err != nil {
		return 0, err
	}
	return resp.Count, nil
}

func (r *RemoteShard) post(ctx context.Context, path string, body, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.baseURL+path, bytes.NewReader(buf))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return decodeResp(resp, r.id, out)
}

func (r *RemoteShard) get(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.baseURL+path, nil)
	if err != nil {
		return err
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return decodeResp(resp, r.id, out)
}

func decodeResp(resp *http.Response, shardID string, out any) error {
	if resp.StatusCode == http.StatusNotFound {
		return store.ErrNotFound
	}
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("remote shard %s: status %d: %s", shardID, resp.StatusCode, string(b))
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

var _ Shard = (*RemoteShard)(nil)
