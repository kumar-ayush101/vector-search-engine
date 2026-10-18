// Package cluster adds horizontal scaling to the engine. Vectors are
// partitioned across shards by a consistent-hash router; a coordinator fans a
// query out to every shard (scatter) and merges the per-shard top-k results
// into a single global top-k (gather). Shards may be in-process (local) or on
// remote nodes reached over HTTP, so the same code path serves a single
// process and a multi-node cluster.
package cluster

import (
	"hash/fnv"
	"sort"
)

// Ring is a consistent-hash ring mapping keys to shard indices. Each shard is
// placed at several virtual points on the ring to keep the key distribution
// even; a key is owned by the first shard clockwise from its hash. Consistent
// hashing means adding or removing a shard only remaps keys near the changed
// shard, not the whole keyspace.
type Ring struct {
	replicas int
	points   []ringPoint // sorted by hash
	numShards int
}

type ringPoint struct {
	hash  uint32
	shard int
}

// NewRing builds a ring over numShards shards with the given number of virtual
// nodes per shard (replicas). More replicas means a smoother distribution.
func NewRing(numShards, replicas int) *Ring {
	if replicas <= 0 {
		replicas = 128
	}
	r := &Ring{replicas: replicas, numShards: numShards}
	for s := 0; s < numShards; s++ {
		for v := 0; v < replicas; v++ {
			r.points = append(r.points, ringPoint{
				hash:  hashKey(vnodeKey(s, v)),
				shard: s,
			})
		}
	}
	sort.Slice(r.points, func(i, j int) bool { return r.points[i].hash < r.points[j].hash })
	return r
}

// Shards returns the number of shards the ring spans.
func (r *Ring) Shards() int { return r.numShards }

// Owner returns the shard index that owns the given key.
func (r *Ring) Owner(key string) int {
	if len(r.points) == 0 {
		return 0
	}
	h := hashKey(key)
	// Binary search for the first point with hash >= h; wrap around if none.
	i := sort.Search(len(r.points), func(i int) bool { return r.points[i].hash >= h })
	if i == len(r.points) {
		i = 0
	}
	return r.points[i].shard
}

// Distribution counts how many of the given keys land on each shard. Useful
// for tests and diagnostics.
func (r *Ring) Distribution(keys []string) []int {
	counts := make([]int, r.numShards)
	for _, k := range keys {
		counts[r.Owner(k)]++
	}
	return counts
}

func vnodeKey(shard, vnode int) string {
	// Distinct, stable string per virtual node.
	return "shard-" + itoa(shard) + "#" + itoa(vnode)
}

func hashKey(s string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(s))
	return h.Sum32()
}

// itoa is a tiny int->string without importing strconv at call sites.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
