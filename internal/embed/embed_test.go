package embed

import (
	"context"
	"testing"

	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

func TestHashEmbedderDeterministic(t *testing.T) {
	e := NewHashEmbedder(64)
	ctx := context.Background()
	a, err := e.Embed(ctx, "the quick brown fox")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	b, _ := e.Embed(ctx, "the quick brown fox")
	if len(a) != 64 {
		t.Fatalf("dim = %d, want 64", len(a))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("embedding not deterministic at %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestHashEmbedderNormalized(t *testing.T) {
	e := NewHashEmbedder(32)
	v, _ := e.Embed(context.Background(), "hello world vectors")
	n := vmath.Norm(v)
	if n < 0.99 || n > 1.01 {
		t.Fatalf("norm = %v, want ~1", n)
	}
}

func TestHashEmbedderEmptyInput(t *testing.T) {
	e := NewHashEmbedder(16)
	v, err := e.Embed(context.Background(), "   ")
	if err != nil {
		t.Fatalf("Embed empty: %v", err)
	}
	if vmath.Norm(v) < 0.99 {
		t.Fatalf("empty input should still yield a unit vector, norm=%v", vmath.Norm(v))
	}
}

func TestHashEmbedderSimilarity(t *testing.T) {
	e := NewHashEmbedder(256)
	ctx := context.Background()
	// Shared tokens should give higher similarity than disjoint ones.
	a, _ := e.Embed(ctx, "machine learning models")
	b, _ := e.Embed(ctx, "machine learning systems")
	c, _ := e.Embed(ctx, "zzz qqq www")
	simAB := vmath.CosineSimilarity(a, b)
	simAC := vmath.CosineSimilarity(a, c)
	if simAB <= simAC {
		t.Fatalf("expected overlapping text more similar: simAB=%v simAC=%v", simAB, simAC)
	}
}

func TestProviderInterface(t *testing.T) {
	var p Provider = NewHashEmbedder(8)
	if p.Dim() != 8 || p.Name() != "hash-local" {
		t.Fatalf("provider metadata wrong: dim=%d name=%q", p.Dim(), p.Name())
	}
}
