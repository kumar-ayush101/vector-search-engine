package hnsw

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/kumar-ayush101/vector-search-engine/internal/vmath"
)

// Snapshot format (little-endian):
//
//	magic   uint32  = 0x56534E31 ("VSN1")
//	version uint16  = 1
//	dim     uint32
//	metric  uint8
//	m       uint16
//	efC     uint16
//	efS     uint16
//	seed    int64
//	entry   uint32
//	maxLvl  uint32
//	hasEntry uint8
//	count   uint32              (number of nodes that follow)
//	repeated node:
//	    id uint32
//	    deleted uint8
//	    vec: dim * float32
//	    nLevels uint8
//	    repeated level: nLinks uint16, links[nLinks] uint32
//
// Writes go to a temp file that is fsynced and atomically renamed, so a
// crash mid-write never leaves a half-written snapshot under the real name.

const (
	snapMagic   uint32 = 0x56534E31
	snapVersion uint16 = 1
)

// Save writes the index to path atomically (temp file + fsync + rename).
func (ix *Index) Save(path string) error {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)

	if err := ix.writeTo(w); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := w.Flush(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

func (ix *Index) writeTo(w io.Writer) error {
	var buf [8]byte
	put32 := func(v uint32) error {
		binary.LittleEndian.PutUint32(buf[:4], v)
		_, err := w.Write(buf[:4])
		return err
	}
	put16 := func(v uint16) error {
		binary.LittleEndian.PutUint16(buf[:2], v)
		_, err := w.Write(buf[:2])
		return err
	}
	put8 := func(v uint8) error {
		buf[0] = v
		_, err := w.Write(buf[:1])
		return err
	}
	put64 := func(v int64) error {
		binary.LittleEndian.PutUint64(buf[:8], uint64(v))
		_, err := w.Write(buf[:8])
		return err
	}

	for _, err := range []error{
		put32(snapMagic), put16(snapVersion), put32(uint32(ix.dim)),
		put8(uint8(ix.metric)), put16(uint16(ix.m)), put16(uint16(ix.efConstr)),
		put16(uint16(ix.efSearch)),
	} {
		if err != nil {
			return err
		}
	}
	if err := put64(ix.seedField()); err != nil {
		return err
	}
	if err := put32(ix.entryPoint); err != nil {
		return err
	}
	if err := put32(uint32(ix.maxLevel)); err != nil {
		return err
	}
	hasEntry := uint8(0)
	if ix.hasEntry {
		hasEntry = 1
	}
	if err := put8(hasEntry); err != nil {
		return err
	}
	if err := put32(uint32(len(ix.nodes))); err != nil {
		return err
	}

	for _, nd := range ix.nodes {
		if err := put32(nd.id); err != nil {
			return err
		}
		del := uint8(0)
		if nd.deleted {
			del = 1
		}
		if err := put8(del); err != nil {
			return err
		}
		for _, f := range nd.vec {
			if err := put32(math.Float32bits(f)); err != nil {
				return err
			}
		}
		if err := put8(uint8(len(nd.links))); err != nil {
			return err
		}
		for _, lvl := range nd.links {
			if err := put16(uint16(len(lvl))); err != nil {
				return err
			}
			for _, id := range lvl {
				if err := put32(id); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// Load reads an index snapshot from path.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readFrom(bufio.NewReader(f))
}

func readFrom(r io.Reader) (*Index, error) {
	var buf [8]byte
	get32 := func() (uint32, error) {
		if _, err := io.ReadFull(r, buf[:4]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(buf[:4]), nil
	}
	get16 := func() (uint16, error) {
		if _, err := io.ReadFull(r, buf[:2]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint16(buf[:2]), nil
	}
	get8 := func() (uint8, error) {
		if _, err := io.ReadFull(r, buf[:1]); err != nil {
			return 0, err
		}
		return buf[0], nil
	}
	get64 := func() (int64, error) {
		if _, err := io.ReadFull(r, buf[:8]); err != nil {
			return 0, err
		}
		return int64(binary.LittleEndian.Uint64(buf[:8])), nil
	}

	magic, err := get32()
	if err != nil {
		return nil, err
	}
	if magic != snapMagic {
		return nil, fmt.Errorf("hnsw: bad snapshot magic %#x", magic)
	}
	ver, err := get16()
	if err != nil {
		return nil, err
	}
	if ver != snapVersion {
		return nil, fmt.Errorf("hnsw: unsupported snapshot version %d", ver)
	}

	dim, _ := get32()
	metricB, _ := get8()
	mB, _ := get16()
	efcB, _ := get16()
	efsB, _ := get16()
	seed, _ := get64()

	ix, err := NewIndex(Params{
		Dim: int(dim), Metric: vmath.Metric(metricB), M: int(mB),
		EfConstruction: int(efcB), EfSearch: int(efsB), Seed: seed,
	})
	if err != nil {
		return nil, err
	}

	entry, _ := get32()
	maxLvl, _ := get32()
	hasEntry, _ := get8()
	count, err := get32()
	if err != nil {
		return nil, err
	}
	ix.entryPoint = entry
	ix.maxLevel = int(maxLvl)
	ix.hasEntry = hasEntry == 1

	for i := uint32(0); i < count; i++ {
		id, err := get32()
		if err != nil {
			return nil, err
		}
		del, _ := get8()
		vec := make([]float32, dim)
		for d := range vec {
			bits, err := get32()
			if err != nil {
				return nil, err
			}
			vec[d] = math.Float32frombits(bits)
		}
		nLevels, _ := get8()
		links := make([][]uint32, nLevels)
		for l := range links {
			nLinks, err := get16()
			if err != nil {
				return nil, err
			}
			lvl := make([]uint32, nLinks)
			for j := range lvl {
				lid, err := get32()
				if err != nil {
					return nil, err
				}
				lvl[j] = lid
			}
			links[l] = lvl
		}
		ix.nodes[id] = &node{id: id, vec: vec, links: links, deleted: del == 1}
	}
	return ix, nil
}

// SaveDir ensures the parent dir of path exists, then saves.
func (ix *Index) SaveDir(path string) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return ix.Save(path)
}
