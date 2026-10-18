package vmath

import (
	"math"
	"testing"
)

func approx(a, b, eps float32) bool {
	return float32(math.Abs(float64(a-b))) <= eps
}

func TestDot(t *testing.T) {
	got := Dot([]float32{1, 2, 3}, []float32{4, 5, 6})
	if got != 32 {
		t.Fatalf("Dot = %v, want 32", got)
	}
}

func TestL2(t *testing.T) {
	got := L2([]float32{0, 0}, []float32{3, 4})
	if !approx(got, 5, 1e-6) {
		t.Fatalf("L2 = %v, want 5", got)
	}
	if sq := L2Squared([]float32{0, 0}, []float32{3, 4}); !approx(sq, 25, 1e-6) {
		t.Fatalf("L2Squared = %v, want 25", sq)
	}
}

func TestCosine(t *testing.T) {
	// Identical direction -> similarity 1, distance 0.
	if s := CosineSimilarity([]float32{1, 0}, []float32{2, 0}); !approx(s, 1, 1e-6) {
		t.Fatalf("cosine sim parallel = %v, want 1", s)
	}
	// Orthogonal -> similarity 0, distance 1.
	if d := CosineDistance([]float32{1, 0}, []float32{0, 1}); !approx(d, 1, 1e-6) {
		t.Fatalf("cosine dist orthogonal = %v, want 1", d)
	}
	// Opposite -> similarity -1, distance 2.
	if d := CosineDistance([]float32{1, 0}, []float32{-1, 0}); !approx(d, 2, 1e-6) {
		t.Fatalf("cosine dist opposite = %v, want 2", d)
	}
}

func TestCosineZeroVectors(t *testing.T) {
	if s := CosineSimilarity([]float32{0, 0}, []float32{0, 0}); s != 1 {
		t.Fatalf("cosine sim of two zero vectors = %v, want 1", s)
	}
	if s := CosineSimilarity([]float32{0, 0}, []float32{1, 1}); s != 0 {
		t.Fatalf("cosine sim with one zero vector = %v, want 0", s)
	}
}

func TestNormalize(t *testing.T) {
	v := Normalize([]float32{3, 4})
	if !approx(Norm(v), 1, 1e-6) {
		t.Fatalf("normalized norm = %v, want 1", Norm(v))
	}
	// Zero vector is unchanged.
	z := Normalize([]float32{0, 0})
	if z[0] != 0 || z[1] != 0 {
		t.Fatalf("normalized zero vector changed: %v", z)
	}
}

func TestDistanceDispatch(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	if d := Distance(Euclidean, a, b); !approx(d, float32(math.Sqrt2), 1e-6) {
		t.Fatalf("Distance(Euclidean) = %v", d)
	}
	// Dot distance is negative inner product: more similar => smaller.
	if d := Distance(DotProduct, []float32{1, 1}, []float32{1, 1}); !approx(d, -2, 1e-6) {
		t.Fatalf("Distance(DotProduct) = %v, want -2", d)
	}
	if d := Distance(Cosine, a, b); !approx(d, 1, 1e-6) {
		t.Fatalf("Distance(Cosine) = %v, want 1", d)
	}
}

func TestParseMetric(t *testing.T) {
	cases := map[string]Metric{
		"":          Cosine,
		"cosine":    Cosine,
		"l2":        Euclidean,
		"euclidean": Euclidean,
		"dot":       DotProduct,
		"ip":        DotProduct,
	}
	for in, want := range cases {
		got, err := ParseMetric(in)
		if err != nil || got != want {
			t.Fatalf("ParseMetric(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	if _, err := ParseMetric("nope"); err == nil {
		t.Fatal("ParseMetric(nope) expected error")
	}
}
