package hnsw

import "container/heap"

// candidate is a node id paired with its distance to the query.
type candidate struct {
	id   uint32
	dist float32
}

// minHeap orders candidates by ascending distance (closest first). It is used
// as the search frontier: we always expand the nearest unvisited node.
type minHeap []candidate

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)         { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// maxHeap orders candidates by descending distance (farthest first). It is
// used to hold the current best results so the worst one can be dropped in
// O(log n) when a closer candidate is found.
type maxHeap []candidate

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)         { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// peek returns the top element without removing it. Caller must ensure Len>0.
func (h maxHeap) peek() candidate { return h[0] }

var (
	_ heap.Interface = (*minHeap)(nil)
	_ heap.Interface = (*maxHeap)(nil)
)
