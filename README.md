# Vector Search Engine

[![CI](https://github.com/kumar-ayush101/vector-search-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/kumar-ayush101/vector-search-engine/actions/workflows/ci.yml)
![Go](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A vector similarity search engine written from scratch in Go. The core index —
an **HNSW** (Hierarchical Navigable Small World) graph — is built on the Go
standard library only: **no FAISS, hnswlib, Pinecone, or pgvector**. It ships
with on-disk persistence, a JSON REST API, server-side embeddings, Prometheus
metrics, Docker packaging and a race-tested suite.

Vector search is the engine behind semantic search and retrieval-augmented
generation (RAG). Most projects reach for a managed service or a C++ library;
this one implements the index, the storage format, and the serving layer by
hand to show how approximate nearest-neighbour (ANN) search actually works end
to end.

---

## Highlights

- **HNSW from scratch** — layered proximity graph with greedy descent through
  exponentially sparse upper layers, then an ef-bounded best-first search on
  layer 0. Measured **recall@10 = 1.0** against brute force on a 2k×32-dim set.
- **Three distance metrics** — cosine, Euclidean (L2) and dot product, unified
  so the index always minimises distance.
- **Crash-safe persistence** — a compact binary snapshot written via temp file
  + `fsync` + atomic rename, so a crash mid-write never corrupts the index.
- **Text or raw vectors** — upsert/query by a raw `vector` or by `text` that is
  embedded server-side through a pluggable `Provider`. A deterministic local
  hash embedder is the default, so the service runs with **no API keys**.
- **Production hygiene** — structured `slog` logging, request IDs, panic
  recovery, CORS, Prometheus `/metrics`, `/health` and `/ready` probes,
  graceful shutdown with a final snapshot, periodic background snapshots.
- **Single static binary** in a non-root Alpine container; CI runs `vet`,
  build, `test -race` and a Docker build on every push.

---

## Architecture

```
                     ┌─────────────────────────────────────┐
   Client ─────────▶ │  HTTP layer (net/http stdlib)        │
                     │  reqID · logging · metrics · recover │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │  store (collection service layer)    │
                     │  string id ⇄ uint32 · metadata       │
                     │  snapshot / restore                  │
                     └──────────────────┬──────────────────┘
                                        │
                      ┌─────────────────┼──────────────────┐
                      ▼                 ▼                   ▼
               hnsw (ANN index)   vmath (metrics)    embed (Provider)
               layered graph      cosine/L2/dot      hash-local default
               Add/Search/Delete                     (pluggable)
                      │
                      ▼
               binary snapshot on disk (fsync + atomic rename)
```

### Project layout

```text
.
├── cmd/server/            # Entrypoint: wiring + graceful shutdown
├── internal/
│   ├── vmath/             # Distance metrics (cosine, L2, dot)
│   ├── hnsw/              # HNSW index + binary snapshot persistence
│   ├── store/             # Collection service: string ids, metadata
│   ├── embed/             # Embedding Provider + local hash embedder
│   ├── config/            # Env-driven configuration
│   ├── metrics/           # Prometheus collectors
│   └── httpapi/           # Router, handlers, middleware
├── Dockerfile             # Multi-stage: build → tiny non-root runtime
├── docker-compose.yml
├── render.yaml            # One-click Render blueprint
└── .github/workflows/     # CI
```

---

## How HNSW works here

HNSW stacks several proximity graphs. Layer 0 holds every point; each higher
layer keeps an exponentially thinning random subset (level drawn from
`floor(-ln(U) / ln M)`). A search starts at the single entry point on the top
layer and **greedily hops toward the query**, descends a layer, and repeats.
The upper layers behave like an express highway that reaches the query's
neighbourhood in `O(log n)` hops; the dense bottom layer then refines the
result with a best-first search bounded by `efSearch`.

- **Insert** finds an entry point via greedy descent, then on each layer at or
  below the node's level runs an `efConstruction` search and links the node to
  its `M` nearest neighbours (`2M` on layer 0), pruning over-full neighbours
  back to budget by keeping the closest links.
- **Search** mirrors insert's descent, then returns the `k` nearest from a
  bounded candidate set, skipping deleted nodes.
- **Delete** is a soft tombstone filtered from results; replacing an id fully
  rewires the graph.

See `internal/hnsw/hnsw.go` for the commented implementation and
`internal/hnsw/hnsw_test.go` for the brute-force recall test.

---

## API reference

All responses are JSON. A request supplies **either** a raw `vector` **or**
`text` to be embedded server-side.

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/api/vectors` | Upsert `{ id, vector? , text?, metadata? }` |
| `GET` | `/api/vectors/{id}` | Get an item's metadata |
| `DELETE` | `/api/vectors/{id}` | Delete an item |
| `POST` | `/api/query` | k-NN query `{ vector?/text?, k }` → ranked hits |
| `GET` | `/api/stats` | Count, dimensionality, embedder name |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

### Example

```bash
# Index a few documents by text (embedded server-side)
curl -X POST http://localhost:8080/api/vectors \
  -H 'Content-Type: application/json' \
  -d '{"id":"d1","text":"deep learning and neural networks","metadata":{"topic":"ml"}}'

curl -X POST http://localhost:8080/api/vectors \
  -H 'Content-Type: application/json' \
  -d '{"id":"d2","text":"cooking pasta with tomato sauce"}'

# Query by text; nearest first
curl -X POST http://localhost:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"neural network training","k":5}'

# Or query by a raw vector
curl -X POST http://localhost:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"vector":[0.12, 0.03, ...],"k":5}'
```

---

## Configuration

All configuration is environment-driven.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `PORT` / `HTTP_PORT` | `8080` | HTTP listen port |
| `APP_ENV` | `development` | `production` switches logs to JSON |
| `DIM` | `128` | Vector dimensionality |
| `METRIC` | `cosine` | `cosine`, `euclidean`/`l2`, or `dot`/`ip` |
| `HNSW_M` | `16` | Max neighbours per node (upper layers) |
| `HNSW_EF_CONSTRUCTION` | `200` | Candidate list size while building |
| `HNSW_EF_SEARCH` | `64` | Candidate list size while querying |
| `HNSW_SEED` | `42` | RNG seed for reproducible level assignment |
| `DATA_DIR` | `./data` | Directory for index snapshots |
| `SNAPSHOT_EVERY` | `30s` | Periodic snapshot interval (`0` disables) |
| `SHUTDOWN_GRACE` | `10s` | Graceful shutdown timeout |

---

## Getting started

```bash
# Run locally
DIM=128 go run ./cmd/server
# open http://localhost:8080/health

# Or with Docker
docker compose up --build
```

---

## Testing

```bash
go test -race ./...     # unit + handler tests, race detector
go test -run TestRecall -v ./internal/hnsw   # recall vs brute force
```

The suite covers distance maths, HNSW insert/search/delete, **recall against a
brute-force baseline**, snapshot round-trips, the store layer, embeddings, and
the full HTTP stack (including semantic ranking of text queries).

---

## Roadmap

The single-node engine above is complete. Distribution is staged next, reusing
patterns from building a storage engine and a distributed scheduler:

| Stage | Component | State |
|---|---|---|
| W1 | Distance metrics (cosine, L2, dot) | **Done** |
| W2 | HNSW index: insert, k-NN search, delete | **Done** |
| W3 | Crash-safe binary snapshot persistence | **Done** |
| W4 | Collection service: string ids, metadata | **Done** |
| W5 | REST API, middleware, health/ready, metrics | **Done** |
| W6 | Server-side embeddings (pluggable provider) | **Done** |
| W7 | Docker, CI, graceful shutdown, periodic snapshots | **Done** |
| P2 | Full metadata durability in the snapshot format | Planned |
| P2 | Sharding: partition vectors across nodes, scatter-gather queries | Planned |
| P2 | Replication of shards for fault tolerance | Planned |
| P3 | Real embedding providers (OpenAI/Cohere/local ONNX) | Planned |
| P3 | Product quantization for memory-efficient vectors | Stretch |

> Status is honest: everything marked **Done** has tests you can run. The
> metadata map is currently rebuilt rather than persisted on restore (the
> vector graph itself is fully durable); closing that gap is the first P2 item.

---

## Tech stack

Go 1.25 · net/http (stdlib routing) · Prometheus · Docker · GitHub Actions

## License

MIT — see [LICENSE](./LICENSE).
