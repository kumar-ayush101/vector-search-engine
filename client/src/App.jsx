import { useState, useEffect, useCallback } from "react";
import { api } from "./api.js";

const SAMPLE_DOCS = [
  { id: "ml-1", text: "deep learning and neural network architectures" },
  { id: "ml-2", text: "training transformers for natural language processing" },
  { id: "db-1", text: "log structured merge trees power write heavy databases" },
  { id: "db-2", text: "distributed consensus with the raft algorithm" },
  { id: "food-1", text: "a classic italian pasta recipe with tomato and basil" },
  { id: "travel-1", text: "backpacking through the mountains of nepal" },
];

export default function App() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");

  const refreshStats = useCallback(async () => {
    try {
      setStats(await api.stats());
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refreshStats();
    const t = setInterval(refreshStats, 3000);
    return () => clearInterval(t);
  }, [refreshStats]);

  const flash = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">◈</span>
          <div>
            <h1>Vector Search Engine</h1>
            <p className="tagline">HNSW index · built from scratch in Go</p>
          </div>
        </div>
        <StatsBar stats={stats} />
      </header>

      {error && <div className="banner error">{error}</div>}
      {toast && <div className="banner toast">{toast}</div>}

      <main className="grid">
        <IndexPanel onChange={refreshStats} flash={flash} setError={setError} />
        <SearchPanel flash={flash} setError={setError} />
      </main>

      <footer className="footer">
        <span>Semantic search demo. Text is embedded server-side; no API keys.</span>
      </footer>
    </div>
  );
}

function StatsBar({ stats }) {
  return (
    <div className="statsbar">
      <Stat label="Vectors" value={stats ? stats.count : "—"} />
      <Stat label="Dimensions" value={stats ? stats.dim : "—"} />
      <Stat label="Embedder" value={stats ? stats.embedder : "—"} />
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function IndexPanel({ onChange, flash, setError }) {
  const [id, setId] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  const upsert = async (docId, docText) => {
    setBusy(true);
    setError("");
    try {
      await api.upsertText(docId, docText, { preview: docText.slice(0, 60) });
      flash(`Indexed "${docId}"`);
      onChange();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const submit = (e) => {
    e.preventDefault();
    if (!id.trim() || !text.trim()) return;
    upsert(id.trim(), text.trim());
    setId("");
    setText("");
  };

  const seed = async () => {
    for (const d of SAMPLE_DOCS) await upsert(d.id, d.text);
    flash("Seeded sample documents");
  };

  return (
    <section className="panel">
      <h2>Index a document</h2>
      <form onSubmit={submit} className="form">
        <input
          className="input"
          placeholder="id (e.g. doc-1)"
          value={id}
          onChange={(e) => setId(e.target.value)}
        />
        <textarea
          className="input textarea"
          placeholder="text to embed and index…"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="row">
          <button className="btn primary" disabled={busy} type="submit">
            {busy ? "Indexing…" : "Index"}
          </button>
          <button className="btn ghost" disabled={busy} type="button" onClick={seed}>
            Seed samples
          </button>
        </div>
      </form>
    </section>
  );
}

function SearchPanel({ flash, setError }) {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [hits, setHits] = useState([]);
  const [busy, setBusy] = useState(false);
  const [searched, setSearched] = useState(false);

  const run = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError("");
    try {
      const res = await api.query(query.trim(), Number(k));
      setHits(res.hits || []);
      setSearched(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id) => {
    try {
      await api.remove(id);
      setHits((h) => h.filter((x) => x.id !== id));
      flash(`Deleted "${id}"`);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section className="panel">
      <h2>Semantic search</h2>
      <form onSubmit={run} className="form">
        <input
          className="input"
          placeholder="search by meaning, e.g. 'neural networks'"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="row">
          <label className="klabel">
            top-k
            <input
              className="input knum"
              type="number"
              min="1"
              max="50"
              value={k}
              onChange={(e) => setK(e.target.value)}
            />
          </label>
          <button className="btn primary" disabled={busy} type="submit">
            {busy ? "Searching…" : "Search"}
          </button>
        </div>
      </form>

      <div className="results">
        {searched && hits.length === 0 && <p className="muted">No results.</p>}
        {hits.map((h, i) => (
          <div className="hit" key={h.id}>
            <div className="hit-rank">#{i + 1}</div>
            <div className="hit-body">
              <div className="hit-id">{h.id}</div>
              {h.metadata && h.metadata.preview && (
                <div className="hit-preview">{h.metadata.preview}</div>
              )}
              <div className="hit-score">
                <ScoreBar score={h.score} />
                <span className="score-num">{h.score.toFixed(3)}</span>
              </div>
            </div>
            <button className="btn tiny" onClick={() => remove(h.id)} title="delete">
              ✕
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}

function ScoreBar({ score }) {
  // Cosine score is in [-1, 1]; map to a 0–100% width.
  const pct = Math.max(0, Math.min(100, ((score + 1) / 2) * 100));
  return (
    <div className="scorebar">
      <div className="scorebar-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}
