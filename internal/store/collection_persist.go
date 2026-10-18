package store

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
)

// metaSnapshot is the sidecar persisted alongside the HNSW index snapshot. It
// captures the string-id <-> internal-id bijection and per-item metadata that
// the index itself does not store, so a restart restores the collection fully
// (not just the vector graph).
type metaSnapshot struct {
	Version  int               `json:"version"`
	Name     string            `json:"name"`
	NextID   uint32            `json:"next_id"`
	Items    []metaItem        `json:"items"`
}

type metaItem struct {
	UID      uint32            `json:"uid"`
	ID       string            `json:"id"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// exportMeta captures the id maps under a read lock.
func (c *Collection) exportMeta() metaSnapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	snap := metaSnapshot{Version: 1, Name: c.name, NextID: c.nextID}
	for s, uid := range c.byString {
		it := c.byUint[uid]
		var md map[string]string
		if it != nil {
			md = cloneMeta(it.Metadata)
		}
		snap.Items = append(snap.Items, metaItem{UID: uid, ID: s, Metadata: md})
	}
	return snap
}

// importMeta restores the id maps from a snapshot under the write lock.
func (c *Collection) importMeta(snap metaSnapshot) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nextID = snap.NextID
	c.byString = make(map[string]uint32, len(snap.Items))
	c.byUint = make(map[uint32]*Item, len(snap.Items))
	for _, it := range snap.Items {
		c.byString[it.ID] = it.UID
		stored := Item{ID: it.ID, Metadata: cloneMeta(it.Metadata)}
		c.byUint[it.UID] = &stored
	}
}

// saveMeta writes the sidecar metadata file atomically (temp + fsync + rename).
func (c *Collection) saveMeta(path string) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	snap := c.exportMeta()
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(snap); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := w.Flush(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// loadMeta reads the sidecar metadata file and restores the id maps. Missing
// file is reported via os.IsNotExist on the returned error.
func (c *Collection) loadMeta(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var snap metaSnapshot
	if err := json.Unmarshal(b, &snap); err != nil {
		return err
	}
	c.importMeta(snap)
	return nil
}
