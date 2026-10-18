// Package config loads runtime configuration from environment variables with
// sensible defaults, so the same binary runs locally and on a managed platform
// without code changes.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

// Config holds all runtime settings.
type Config struct {
	Env           string        // "development" or "production"
	HTTPPort      string        // listen port
	DataDir       string        // where index snapshots are written
	Dim           int           // vector dimensionality
	Metric        vmath.Metric  // distance metric
	M             int           // HNSW M
	EfConstruction int          // HNSW build-time candidate list
	EfSearch      int           // HNSW query-time candidate list
	Seed          int64         // RNG seed
	SnapshotEvery time.Duration // periodic snapshot interval (0 = disabled)
	ShutdownGrace time.Duration // graceful shutdown timeout
}

// Load reads configuration from the environment.
func Load() (*Config, error) {
	metric, err := vmath.ParseMetric(getenv("METRIC", "cosine"))
	if err != nil {
		return nil, err
	}
	cfg := &Config{
		Env:            getenv("APP_ENV", "development"),
		HTTPPort:       firstNonEmpty(os.Getenv("PORT"), os.Getenv("HTTP_PORT"), "8080"),
		DataDir:        getenv("DATA_DIR", "./data"),
		Dim:            getenvInt("DIM", 128),
		Metric:         metric,
		M:              getenvInt("HNSW_M", 16),
		EfConstruction: getenvInt("HNSW_EF_CONSTRUCTION", 200),
		EfSearch:       getenvInt("HNSW_EF_SEARCH", 64),
		Seed:           int64(getenvInt("HNSW_SEED", 42)),
		SnapshotEvery:  getenvDuration("SNAPSHOT_EVERY", 30*time.Second),
		ShutdownGrace:  getenvDuration("SHUTDOWN_GRACE", 10*time.Second),
	}
	if cfg.Dim <= 0 {
		return nil, fmt.Errorf("config: DIM must be > 0, got %d", cfg.Dim)
	}
	return cfg, nil
}

// IsProduction reports whether the app runs in production mode.
func (c *Config) IsProduction() bool { return c.Env == "production" }

// SnapshotPath is the full path of the index snapshot file.
func (c *Config) SnapshotPath() string {
	return c.DataDir + "/index.vsnap"
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getenvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func getenvDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
