# Vector Search Engine

[![CI](https://github.com/kumar-ayush101/vector-search-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/kumar-ayush101/vector-search-engine/actions/workflows/ci.yml)
![Go](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Live demo:** https://vector-search-engine.onrender.com
> (single-node instance on a free tier — the first request after idle may take
> a moment to wake, and demo data resets when the instance restarts)

A **distributed** vector similarity search engine written from scratch in Go.
The core index — an **HNSW** (Hierarchical Navigable Small World) graph — uses
only the Go standard library: **no FAISS, hnswlib, Pinecone, or pgvector**. On
top of the index it adds crash-safe persistence, consistent-hash **sharding
with scatter-gather queries across a multi-node cluster**, server-side
embeddings, Prometheus metrics, a React dashboard served from the same binary,
and a measured benchmark harness.

Vector search is the engine behind semantic search and retrieval-augmented
generation (RAG). Most projects reach for a managed service or a C++ library;
this one implements the index, the storage format, the distribution layer and
the serving layer by hand.

---

## Highlights

- **HNSW from scratch** — layered proximity graph with greedy descent through
  exponentially sparse upper layers, then an `ef`-bounded best-first search on
  layer 0. Measured **recall@10 up to 0.97** against brute force (see below).
- **Distributed scatter-gather** — vectors are partitioned across shards by a
  **consistent-hash ring**; a coordinator fans each query out to all shards in
  parallel and merges a global top-k with a bounded heap. Sharded results match
  the single-node baseline with **0.999 top-k agreement**.
- **Multi-node cluster** — shards can live in other processes, reached over
  HTTP. The same binary runs standalone or as a cluster node (set `PEERS`).
- **Crash-safe persistence** — a compact binary snapshot of the graph plus a
  metadata sidecar, both written via temp file + `fsync` + atomic rename, so a
  restart restores vectors **and** ids/metadata intact.
- **Text or raw vectors** — upsert/query by a raw `vector` or by `text`
  embedded server-side through a pluggable `Provider`. A deterministic local
  embedder is the default, so the service runs with **no API keys**.
- **React dashboard** served from the Go binary (single origin): index
  documents, run semantic search, watch live stats.
- **Production hygiene** — structured `slog` logs, request IDs, panic recovery,
  CORS, Prometheus `/metrics`, `/health` + `/ready`, graceful shutdown with a
  final snapshot, periodic background snapshots. CI runs vet, build,
  `test -race`, a frontend build, and a Docker build.

---

## Architecture

```
                     ┌─────────────────────────────────────┐
   Browser / API ──▶ │  HTTP layer (net/http stdlib)        │
                     │  reqID · logging · metrics · recover │
                     │  + React dashboard (single origin)   │
                     └──────────────────┬──────────────────┘
                                        │  Engine interface
                        ┌───────────────┴───────────────┐
                        ▼                                ▼
                single-node store              cluster Coordinator
                (one HNSW index)               consistent-hash ring
                        │                        │  scatter / gather
                        │                 ┌──────┼───────┐
                        ▼                 ▼      ▼       ▼
                  hnsw + vmath         shard0  shard1  shard2
                  snapshot + meta      (local) (remote over HTTP)
```

### Project layout

```text
.
├── cmd/
│   ├── server/            # Entrypoint: single-node or cluster
│   └── vsbench/           # Benchmark harness (recall / latency / QPS)
├── internal/
│   ├── vmath/             # Distance metrics (cosine, L2, dot)
│   ├── hnsw/              # HNSW index + binary snapshot persistence
│   ├── store/             # Collection: string ids, metadata, durable snapshot
│   ├── cluster/           # Ring, shards, coordinator, remote shard over HTTP
│   ├── embed/             # Embedding Provider + local hash embedder
│   ├── config/            # Env-driven configuration
│   ├── metrics/           # Prometheus collectors
│   └── httpapi/           # Router, handlers, middleware, Engine interface
├── client/                # React (Vite) dashboard → builds to ./web
├── Dockerfile             # 3-stage: frontend → Go → tiny runtime
├── docker-compose.yml     # Local 3-node cluster
└── render.yaml            # One-click Render blueprint
```

---

## Benchmarks

`go run ./cmd/vsbench -n 5000 -dim 128 -k 10 -queries 500` sweeps `efSearch`
and measures recall@k against a brute-force ground truth, plus latency and
throughput. Representative run (Go 1.25, single machine — numbers are relative
and depend on CPU):

| efSearch | recall@10 | p99 latency | QPS |
|---|---|---|---|
| 10  | 0.44 | ~1.1 ms | ~10100 |
| 20  | 0.60 | ~1.5 ms | ~5300  |
| 40  | 0.77 | ~1.8 ms | ~3200  |
| 64  | 0.86 | ~1.7 ms | ~2500  |
| 100 | 0.93 | ~2.6 ms | ~1700  |
| 200 | 0.97 | ~3.7 ms | ~1100  |

This is the fundamental ANN tradeoff: a larger `efSearch` widens the search
frontier, raising recall at the cost of latency and throughput.

---

## How HNSW works here

HNSW stacks several proximity graphs. Layer 0 holds every point; each higher
layer keeps an exponentially thinning random subset. A search starts at the
entry point on the top layer, **greedily hops toward the query**, descends a
layer, and repeats; the upper layers reach the query's neighbourhood in
`O(log n)` hops and the dense bottom layer refines the result with a best-first
search bounded by `efSearch`. Insert links each node to its `M` nearest
neighbours per layer (`2M` on layer 0), pruning over-full neighbours back to
budget. See `internal/hnsw/hnsw.go`.

## How sharding works here

A consistent-hash ring (`internal/cluster/ring.go`) maps each vector id to an
owning shard using virtual nodes, so adding or removing a shard only remaps
keys near the change. Writes route to the owning shard; **queries scatter to
every shard in parallel and the coordinator gathers the global top-k** with a
max-heap of size k (`internal/cluster/coordinator.go`). Shards implement a
single `Shard` interface, so an in-process `LocalShard` and an HTTP-backed
`RemoteShard` are interchangeable — the same code path serves one process or a
cluster.

---

## API reference

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/api/vectors` | Upsert `{ id, vector? , text?, metadata? }` |
| `GET` | `/api/vectors/{id}` | Get an item's metadata |
| `DELETE` | `/api/vectors/{id}` | Delete an item |
| `POST` | `/api/query` | k-NN query `{ vector?/text?, k }` → ranked hits |
| `GET` | `/api/stats` | Count, dimensionality, embedder name |
| `GET` | `/health` · `/ready` | Probes |
| `GET` | `/metrics` | Prometheus metrics |

```bash
curl -X POST http://localhost:8080/api/vectors \
  -H 'Content-Type: application/json' \
  -d '{"id":"d1","text":"deep learning and neural networks"}'

curl -X POST http://localhost:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"neural network training","k":5}'
```

---

## Configuration

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `PORT` / `HTTP_PORT` | `8080` | HTTP listen port |
| `APP_ENV` | `development` | `production` switches logs to JSON |
| `DIM` | `128` | Vector dimensionality |
| `METRIC` | `cosine` | `cosine`, `euclidean`/`l2`, or `dot`/`ip` |
| `HNSW_M` | `16` | Max neighbours per node (upper layers) |
| `HNSW_EF_CONSTRUCTION` | `200` | Candidate list size while building |
| `HNSW_EF_SEARCH` | `64` | Candidate list size while querying |
| `DATA_DIR` | `./data` | Directory for index snapshots |
| `SNAPSHOT_EVERY` | `30s` | Periodic snapshot interval (`0` disables) |
| `PEERS` | — | CSV of peer URLs; two or more enables cluster mode |
| `NODE_INDEX` | `0` | This node's position in `PEERS` |
| `SERVE_FRONTEND` | `true` | Serve the React dashboard |
| `FRONTEND_DIR` | `./web` | Static assets directory |

---

## Getting started

### Single node (API + dashboard)

```bash
cd client && npm install && npm run build && cd ..   # build the dashboard
DIM=128 go run ./cmd/server                          # http://localhost:8080
```

### Multi-node cluster via Docker Compose

```bash
docker compose up --build          # 3 nodes form a sharded cluster
# open http://localhost:8080 — query any node, it scatter-gathers across all
```

### Frontend dev server

```bash
cd client && npm run dev           # http://localhost:5173, proxies to :8080
```

---

## Testing

```bash
go test -race ./...                               # unit + handler + cluster
go run ./cmd/vsbench                               # benchmark table
go test -run TestScatterGather -v ./internal/cluster   # sharding correctness
```

The suite covers distance maths, HNSW insert/search/delete, **recall vs
brute-force**, snapshot round-trips, **metadata durability across restart**,
the store layer, embeddings, the full HTTP stack, consistent-hash distribution,
**scatter-gather vs single-node agreement**, and a **real multi-node cluster
over HTTP**.

---

## Roadmap

| Stage | Component | State |
|---|---|---|
| W1 | Distance metrics (cosine, L2, dot) | **Done** |
| W2 | HNSW index: insert, k-NN search, delete | **Done** |
| W3 | Crash-safe snapshot persistence (graph + metadata) | **Done** |
| W4 | Collection service: string ids, metadata | **Done** |
| W5 | REST API, middleware, health/ready, metrics | **Done** |
| W6 | Server-side embeddings (pluggable provider) | **Done** |
| W7 | Benchmark harness with measured recall/latency/QPS | **Done** |
| W8 | Consistent-hash sharding + scatter-gather queries | **Done** |
| W9 | Multi-node cluster (remote shards over HTTP) | **Done** |
| W10 | React dashboard served from the Go binary | **Done** |
| W11 | Docker (multi-stage), compose cluster, CI, Render | **Done** |
| P2 | Shard replication for fault tolerance | Planned |
| P3 | Real embedding providers (OpenAI/Cohere/local ONNX) | Planned |
| P3 | Product quantization for memory-efficient vectors | Stretch |

> Everything marked **Done** has tests you can run.

## Tech stack

Go 1.25 · net/http (stdlib routing) · React 18 + Vite · Prometheus · Docker ·
GitHub Actions

## License

MIT — see [LICENSE](./LICENSE).
