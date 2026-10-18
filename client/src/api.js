// Thin fetch wrapper around the vector search REST API. Same-origin in
// production (served by the Go binary), proxied to :8080 in dev.

async function jsonFetch(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.error) msg = body.error;
    } catch {
      /* non-JSON error */
    }
    throw new Error(msg);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  stats: () => jsonFetch("/api/stats"),
  upsertText: (id, text, metadata) =>
    jsonFetch("/api/vectors", {
      method: "POST",
      body: JSON.stringify({ id, text, metadata }),
    }),
  query: (text, k) =>
    jsonFetch("/api/query", {
      method: "POST",
      body: JSON.stringify({ text, k }),
    }),
  remove: (id) => jsonFetch(`/api/vectors/${encodeURIComponent(id)}`, { method: "DELETE" }),
};
