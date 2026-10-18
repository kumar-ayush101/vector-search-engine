// Package vmath provides distance and similarity functions over float32
// vectors. These are the primitives the index uses to compare embeddings.
//
// All functions assume the two input slices have equal length; callers are
// expected to validate dimensionality before calling (the index does this
// once at insert time). Keeping these functions allocation-free and
// branch-light matters because they run on every distance comparison during a
// search, which dominates query cost.
package vmath

import (
	"errors"
	"math"
)

// ErrDimMismatch is returned by helpers that validate dimensionality.
var ErrDimMismatch = errors.New("vmath: vector dimension mismatch")

// Metric identifies a distance function. Smaller distance means more similar.
type Metric uint8

const (
	// Cosine distance: 1 - cosine similarity. Range [0, 2].
	Cosine Metric = iota
	// Euclidean (L2) distance.
	Euclidean
	// DotProduct distance: negative inner product, so that a larger inner
	// product (more similar) yields a smaller distance.
	DotProduct
)

// String renders the metric name.
func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case Euclidean:
		return "euclidean"
	case DotProduct:
		return "dot"
	default:
		return "unknown"
	}
}

// ParseMetric maps a string to a Metric, defaulting to Cosine for empty input.
func ParseMetric(s string) (Metric, error) {
	switch s {
	case "", "cosine":
		return Cosine, nil
	case "euclidean", "l2":
		return Euclidean, nil
	case "dot", "dotproduct", "ip":
		return DotProduct, nil
	default:
		return Cosine, errors.New("vmath: unknown metric " + s)
	}
}

// Distance computes the distance between a and b under metric m. Smaller is
// more similar for every metric, so the index can always treat it as a
// minimisation problem.
func Distance(m Metric, a, b []float32) float32 {
	switch m {
	case Euclidean:
		return L2(a, b)
	case DotProduct:
		return -Dot(a, b)
	default:
		return CosineDistance(a, b)
	}
}

// Dot returns the inner product of a and b.
func Dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// L2 returns the Euclidean distance between a and b.
func L2(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

// L2Squared returns the squared Euclidean distance. It avoids the sqrt, which
// is enough when only the ordering of distances matters (as in k-NN search).
func L2Squared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// Norm returns the L2 norm (magnitude) of a.
func Norm(a []float32) float32 {
	return float32(math.Sqrt(float64(Dot(a, a))))
}

// CosineSimilarity returns the cosine of the angle between a and b, in
// [-1, 1]. Two zero vectors are defined as perfectly similar (1); a single
// zero vector is defined as orthogonal (0) to avoid NaN.
func CosineSimilarity(a, b []float32) float32 {
	na := Norm(a)
	nb := Norm(b)
	if na == 0 && nb == 0 {
		return 1
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return Dot(a, b) / (na * nb)
}

// CosineDistance returns 1 - cosine similarity, in [0, 2].
func CosineDistance(a, b []float32) float32 {
	return 1 - CosineSimilarity(a, b)
}

// Normalize scales a to unit length in place and returns it. A zero vector is
// left unchanged. Normalising up front lets cosine search use a plain dot
// product, which is cheaper per comparison.
func Normalize(a []float32) []float32 {
	n := Norm(a)
	if n == 0 {
		return a
	}
	inv := 1 / n
	for i := range a {
		a[i] *= inv
	}
	return a
}
