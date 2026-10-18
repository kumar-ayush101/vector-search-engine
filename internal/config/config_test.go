package config

import (
	"testing"
	"time"

	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

func TestLoadDefaults(t *testing.T) {
	// Ensure a clean environment.
	for _, k := range []string{"APP_ENV", "PORT", "HTTP_PORT", "DATA_DIR", "DIM", "METRIC", "HNSW_M", "SNAPSHOT_EVERY"} {
		t.Setenv(k, "")
	}
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.HTTPPort != "8080" {
		t.Fatalf("default port = %q, want 8080", cfg.HTTPPort)
	}
	if cfg.Dim != 128 {
		t.Fatalf("default dim = %d, want 128", cfg.Dim)
	}
	if cfg.Metric != vmath.Cosine {
		t.Fatalf("default metric = %v, want cosine", cfg.Metric)
	}
	if cfg.IsProduction() {
		t.Fatal("default should not be production")
	}
}

func TestLoadOverrides(t *testing.T) {
	t.Setenv("APP_ENV", "production")
	t.Setenv("HTTP_PORT", "9000")
	t.Setenv("DIM", "256")
	t.Setenv("METRIC", "l2")
	t.Setenv("SNAPSHOT_EVERY", "1m")
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !cfg.IsProduction() {
		t.Fatal("expected production")
	}
	if cfg.HTTPPort != "9000" {
		t.Fatalf("port = %q, want 9000", cfg.HTTPPort)
	}
	if cfg.Dim != 256 {
		t.Fatalf("dim = %d, want 256", cfg.Dim)
	}
	if cfg.Metric != vmath.Euclidean {
		t.Fatalf("metric = %v, want euclidean", cfg.Metric)
	}
	if cfg.SnapshotEvery != time.Minute {
		t.Fatalf("snapshot every = %v, want 1m", cfg.SnapshotEvery)
	}
}

func TestPortPrefersPORT(t *testing.T) {
	t.Setenv("PORT", "7000")
	t.Setenv("HTTP_PORT", "9000")
	cfg, _ := Load()
	if cfg.HTTPPort != "7000" {
		t.Fatalf("port = %q, want 7000 (PORT takes precedence)", cfg.HTTPPort)
	}
}

func TestInvalidDim(t *testing.T) {
	t.Setenv("DIM", "0")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for DIM=0")
	}
}
