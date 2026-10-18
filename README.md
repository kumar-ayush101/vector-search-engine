# Vector Search Engine

A vector similarity search engine written from scratch in Go, using only the
standard library for the core index — no FAISS, hnswlib, or external ANN
bindings. It implements an **HNSW** (Hierarchical Navigable Small World) graph
for approximate nearest-neighbour search, with on-disk persistence, a REST API,
pluggable embeddings, and Prometheus observability.

> Work in progress. See the roadmap below for staged milestones.

## Why

Vector search powers semantic search and retrieval-augmented generation (RAG).
Most projects reach for a managed service (Pinecone) or a library (FAISS); this
one builds the index, the storage, and the serving layer by hand to show how
approximate nearest-neighbour search actually works end to end.

## Status

See the roadmap in this README as the project is built out in stages.

## License

MIT — see [LICENSE](./LICENSE).
