// Package metrics defines and registers the Prometheus collectors used across
// the service: HTTP traffic, query latency and index size.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Collectors groups the application metrics.
type Collectors struct {
	HTTPRequests  *prometheus.CounterVec
	HTTPDuration  *prometheus.HistogramVec
	QueryDuration prometheus.Histogram
	QueriesTotal  prometheus.Counter
	UpsertsTotal  prometheus.Counter
	IndexSize     prometheus.Gauge
}

// New registers and returns the collectors.
func New() *Collectors {
	return &Collectors{
		HTTPRequests: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total HTTP requests by method, route and status.",
		}, []string{"method", "route", "status"}),
		HTTPDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request latency in seconds.",
			Buckets: prometheus.DefBuckets,
		}, []string{"method", "route"}),
		QueryDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "vector_query_duration_seconds",
			Help:    "Vector k-NN query latency in seconds.",
			Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5},
		}),
		QueriesTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "vector_queries_total",
			Help: "Total number of vector queries served.",
		}),
		UpsertsTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "vector_upserts_total",
			Help: "Total number of vector upserts.",
		}),
		IndexSize: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "vector_index_size",
			Help: "Current number of live vectors in the index.",
		}),
	}
}
