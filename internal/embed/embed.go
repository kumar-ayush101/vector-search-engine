// Package embed turns text into vectors. It defines a provider interface so a
// real model (OpenAI, Cohere, a local ONNX model) can be plugged in, and ships
// a deterministic hashing embedder as the default so the service runs with no
// external dependencies or API keys — useful for demos, tests and CI.
package embed

import (
	"context"
	"hash/fnv"
	"math"
	"strings"

	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

// Provider converts text into a fixed-dimension embedding vector.
type Provider interface {
	// Embed returns a vector for the given text.
	Embed(ctx context.Context, text string) ([]float32, error)
	// Dim is the dimensionality of vectors this provider returns.
	Dim() int
	// Name identifies the provider in logs and health output.
	Name() string
}

// HashEmbedder is a deterministic, dependency-free embedder. It maps text to a
// fixed-dimension vector using the hashing trick: each token is hashed to a
// dimension and a sign, and token counts accumulate there. The vector is then
// L2-normalised so cosine similarity is meaningful.
//
// It is NOT semantic — "car" and "automobile" are unrelated — but it is
// stable, fast and good enough to demonstrate the end-to-end pipeline and to
// make tests reproducible. Swap in a real Provider for semantic quality.
type HashEmbedder struct {
	dim int
}

// NewHashEmbedder returns a HashEmbedder of the given dimensionality.
func NewHashEmbedder(dim int) *HashEmbedder {
	if dim <= 0 {
		dim = 128
	}
	return &HashEmbedder{dim: dim}
}

// Dim implements Provider.
func (h *HashEmbedder) Dim() int { return h.dim }

// Name implements Provider.
func (h *HashEmbedder) Name() string { return "hash-local" }

// Embed implements Provider. The context is accepted for interface symmetry
// with network-backed providers but is not used here.
func (h *HashEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	vec := make([]float32, h.dim)
	for _, tok := range tokenize(text) {
		idx, sign := bucket(tok, h.dim)
		vec[idx] += sign
	}
	// Guard against an all-zero vector (empty or whitespace-only input): a
	// single stable bump keeps cosine distance well defined.
	if isZero(vec) {
		vec[0] = 1
	}
	vmath.Normalize(vec)
	return vec, nil
}

// tokenize lowercases and splits text on non-alphanumeric runs.
func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	})
}

// bucket hashes a token to a dimension index and a +1/-1 sign (a second hash
// bit), the standard signed hashing trick that reduces collision bias.
func bucket(tok string, dim int) (int, float32) {
	hh := fnv.New32a()
	_, _ = hh.Write([]byte(tok))
	sum := hh.Sum32()
	idx := int(sum % uint32(dim))
	sign := float32(1)
	if sum&0x80000000 != 0 {
		sign = -1
	}
	return idx, sign
}

func isZero(v []float32) bool {
	for _, x := range v {
		if math.Abs(float64(x)) > 1e-12 {
			return false
		}
	}
	return true
}

// Ensure HashEmbedder satisfies Provider.
var _ Provider = (*HashEmbedder)(nil)
