import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output goes to ../web so the Go binary can serve it as static assets
// from a single origin. In dev, API calls are proxied to the Go server.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../web",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://localhost:8080",
      "/health": "http://localhost:8080",
      "/ready": "http://localhost:8080",
    },
  },
});
