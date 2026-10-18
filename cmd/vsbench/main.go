// Command vsbench benchmarks the HNSW index: it builds an index over a random
// dataset, then for a sweep of efSearch values measures recall@k against a
// brute-force ground truth, plus query latency (p50/p99) and throughput. It
// prints a Markdown table so results can be pasted straight into the README.
//
// Usage:
//
//	go run ./cmd/vsbench [-n 10000] [-dim 128] [-k 10] [-queries 1000]
//
// All numbers are measured on the machine that runs it and depend heavily on
// CPU and dataset; treat them as relative, not absolute.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"sort"
	"time"

	"github.com/kumar-ayush101/vector-search-engine/internal/hnsw"
	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

func main() {
	n := flag.Int("n", 10000, "number of vectors to index")
	dim := flag.Int("dim", 128, "vector dimensionality")
	k := flag.Int("k", 10, "k for recall@k")
	queries := flag.Int("queries", 1000, "number of query vectors")
	m := flag.Int("m", 16, "HNSW M")
	efc := flag.Int("efc", 200, "HNSW efConstruction")
	seed := flag.Int64("seed", 42, "RNG seed")
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))
	fmt.Printf("Building index: n=%d dim=%d M=%d efConstruction=%d\n", *n, *dim, *m, *efc)

	// Generate dataset.
	data := make([][]float32, *n)
	for i := range data {
		data[i] = randVec(rng, *dim)
	}

	ix, err := hnsw.NewIndex(hnsw.Params{
		Dim: *dim, Metric: vmath.Euclidean, M: *m, EfConstruction: *efc, EfSearch: 64, Seed: *seed,
	})
	if err != nil {
		panic(err)
	}

	buildStart := time.Now()
	for i, v := range data {
		if err := ix.Add(uint32(i), v); err != nil {
			panic(err)
		}
	}
	buildDur := time.Since(buildStart)
	insRate := float64(*n) / buildDur.Seconds()
	fmt.Printf("Built in %s (%.0f inserts/sec)\n\n", buildDur.Truncate(time.Millisecond), insRate)

	// Query set + brute-force ground truth.
	qs := make([][]float32, *queries)
	truth := make([][]uint32, *queries)
	for i := range qs {
		qs[i] = randVec(rng, *dim)
		truth[i] = bruteForce(data, qs[i], *k)
	}

	fmt.Printf("| efSearch | recall@%d | p50 (µs) | p99 (µs) | QPS |\n", *k)
	fmt.Println("|---|---|---|---|---|")

	for _, ef := range []int{10, 20, 40, 64, 100, 200} {
		ix.SetEfSearch(ef)
		latencies := make([]time.Duration, *queries)
		var hits int
		for i, q := range qs {
			start := time.Now()
			res, err := ix.Search(q, *k)
			latencies[i] = time.Since(start)
			if err != nil {
				panic(err)
			}
			hits += overlap(res, truth[i])
		}
		recall := float64(hits) / float64(*queries**k)
		p50 := percentile(latencies, 50)
		p99 := percentile(latencies, 99)
		qps := float64(*queries) / sumDur(latencies).Seconds()
		fmt.Printf("| %d | %.3f | %d | %d | %.0f |\n",
			ef, recall, p50.Microseconds(), p99.Microseconds(), qps)
	}
}

func randVec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}

func bruteForce(data [][]float32, q []float32, k int) []uint32 {
	type pair struct {
		id uint32
		d  float32
	}
	all := make([]pair, len(data))
	for i, v := range data {
		all[i] = pair{uint32(i), vmath.L2Squared(q, v)}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].d < all[j].d })
	out := make([]uint32, 0, k)
	for i := 0; i < k && i < len(all); i++ {
		out = append(out, all[i].id)
	}
	return out
}

func overlap(res []hnsw.Result, truth []uint32) int {
	set := make(map[uint32]struct{}, len(truth))
	for _, id := range truth {
		set[id] = struct{}{}
	}
	n := 0
	for _, r := range res {
		if _, ok := set[r.ID]; ok {
			n++
		}
	}
	return n
}

func percentile(ds []time.Duration, p int) time.Duration {
	cp := make([]time.Duration, len(ds))
	copy(cp, ds)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	idx := (p * len(cp)) / 100
	if idx >= len(cp) {
		idx = len(cp) - 1
	}
	return cp[idx]
}

func sumDur(ds []time.Duration) time.Duration {
	var s time.Duration
	for _, d := range ds {
		s += d
	}
	return s
}
