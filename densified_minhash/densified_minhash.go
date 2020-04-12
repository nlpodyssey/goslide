// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package densified_minhash

import (
	"container/heap"
	"encoding/binary"
	"hash/maphash"
	"math"
	"math/rand"
	"time"
)

type DensifiedMinhash struct {
	randHash   [2]int
	randa      int
	numHashes  int
	rangePow   int
	logNumHash int
	seed       maphash.Seed
}

func New(numHashes, numOfBitsToHash int) *DensifiedMinhash {
	rand.Seed(time.Now().UnixNano())

	return &DensifiedMinhash{
		// TODO: is second value of `randHash` used at all?
		randHash:   [2]int{positiveOddRandomInt(), positiveOddRandomInt()},
		randa:      positiveOddRandomInt(),
		numHashes:  numHashes,
		rangePow:   numOfBitsToHash,
		logNumHash: int(math.Log2(float64(numHashes))),
		seed:       maphash.MakeSeed(),
	}
}

func (dm *DensifiedMinhash) GetHash(
	indices []int,
	data []float64,
	binIds []int,
) []int {
	hashes := make([]int, dm.numHashes)
	hashArray := make([]int, dm.numHashes)

	for i := range hashes {
		hashes[i] = math.MinInt64
	}

	for i := range data {
		binId := binIds[indices[i]]
		if hashes[binId] < indices[i] {
			hashes[binId] = indices[i]
		}
	}

	for i, next := range hashes {
		if next != math.MinInt64 {
			hashArray[i] = next
			continue
		}

		for count := 1; next == math.MinInt64; count++ {
			index := minInt(dm.GetRandDoubleHash(i, count), dm.numHashes)
			next = hashes[index] // Kills GPU.

			if count > 100 { // Densification failure.
				break
			}
		}

		hashArray[i] = next
	}

	return hashArray
}

func (dm *DensifiedMinhash) GetHashEasy(
	binIds []int,
	data []float64,
	topK int,
) []int {
	// Read the data and add it to priority queue O(dlogk approx 7d)
	// with index as key and values as priority value, get topk index
	// O(1) and apply minhash on retuned index.
	pq := make(indexValuePriorityQueue, topK, topK+1)
	for index, value := range data[0:topK] {
		pq[index] = indexValuePair{index, value}
	}
	heap.Init(&pq)

	dataLen := len(data)
	for index := topK; index < dataLen; index++ {
		heap.Push(&pq, indexValuePair{index, data[index]})
		heap.Pop(&pq)
	}

	hashes := make([]int, dm.numHashes)
	hashArray := make([]int, dm.numHashes)

	for i := range hashes {
		hashes[i] = math.MinInt64
	}

	for i := 0; i < topK; i++ {
		pair := pq.Pop().(indexValuePair)
		index := pair.index
		binId := binIds[index]
		if hashes[binId] < index {
			hashes[binId] = index
		}
	}

	for i, next := range hashes {
		if next != math.MinInt64 {
			hashArray[i] = next
			continue
		}

		for count := 1; next == math.MinInt64; count++ {
			index := minInt(dm.GetRandDoubleHash(i, count), dm.numHashes)
			next = hashes[index] // Kills GPU.

			if count > 100 { // Densification failure.
				break
			}
		}

		hashArray[i] = next
	}

	return hashArray
}

func (dm *DensifiedMinhash) GetRandDoubleHash(binId, count int) int {
	toHash := ((binId + 1) << 6) + count
	return (dm.randHash[0] * toHash << 3) >> (32 - dm.logNumHash) // logNumHash needs to be ceiled.
}

func (dm *DensifiedMinhash) GetMap(n int) []int {
	binIds := make([]int, n)

	rng := 1 << dm.rangePow

	// The number of times the range is larger than
	// the total number of hashes we need.
	binSize := int(math.Ceil(float64(rng) / float64(dm.numHashes)))

	buf := make([]byte, 8)

	mh := maphash.Hash{}
	mh.SetSeed(dm.seed)

	for i := 0; i < n; i++ {
		bufLen := binary.PutVarint(buf, int64(i))
		mh.Reset()
		mh.Write(buf[:bufLen])
		curHash := mh.Sum64()

		curHash = curHash & ((1 << dm.rangePow) - 1)
		binIds[i] = int(math.Floor(float64(curHash) / float64(binSize)))
	}

	return binIds
}

func positiveOddRandomInt() int {
	n := rand.Int()
	if n%2 == 0 {
		return n + 1
	}
	return n
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
