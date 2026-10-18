# syntax=docker/dockerfile:1

# ---- Stage 1: build the React dashboard ----
FROM node:20-alpine AS frontend
WORKDIR /ui
COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/ ./
RUN npm run build   # outputs to /web (vite outDir ../web)

# ---- Stage 2: build the Go binary ----
FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/server ./cmd/server

# ---- Stage 3: tiny runtime ----
FROM alpine:3.20
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app

COPY --from=build /out/server /app/server
COPY --from=frontend /web /app/web

RUN mkdir -p /app/data && chown -R app:app /app
USER app

ENV DATA_DIR=/app/data
ENV FRONTEND_DIR=/app/web
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/server"]
