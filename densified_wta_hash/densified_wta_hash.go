// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of Densified Winner Take All (WTA) hash
//
// Algorithm from the paper:
//   Densified Winner Take All (WTA) Hashing for Sparse Datasets
//   Beidi Chen, Anshumali Shrivastava
//   http://auai.org/uai2018/proceedings/papers/321.pdf
package densified_wta_hash

import (
	"math"
	"math/rand"
	"time"
)

// The number of times the range is larger than
// the total number of hashes we need.
const binSize = 8

type DensifiedWtaHash struct {
	randHash   [2]int
	randa      int
	numHashes  int
	rangePow   int
	logNumHash int
	indices    []int
	pos        []int
	permute    int
}

func New(numHashes, numOfBitsToHash int) *DensifiedWtaHash {
	rand.Seed(time.Now().UnixNano())

	permute := int(math.Ceil(
		float64(numHashes) * float64(binSize) / float64(numOfBitsToHash)))
	nArray := make([]int, numOfBitsToHash)
	indices := make([]int, numOfBitsToHash*permute)
	pos := make([]int, numOfBitsToHash*permute)

	for i := range nArray {
		nArray[i] = i
	}

	swap := func(i, j int) {
		nArray[i], nArray[j] = nArray[j], nArray[i]
	}

	for p := 0; p < permute; p++ {
		rand.Shuffle(numOfBitsToHash, swap)

		for j, value := range nArray {
			indices[p*numOfBitsToHash+value] = (p*numOfBitsToHash + j) / binSize
			pos[p*numOfBitsToHash+value] = (p*numOfBitsToHash + j) % binSize
		}
	}

	return &DensifiedWtaHash{
		// TODO: is second value of `randHash` used at all?
		randHash:   [2]int{positiveOddRandomInt(), positiveOddRandomInt()},
		randa:      positiveOddRandomInt(),
		numHashes:  numHashes,
		rangePow:   numOfBitsToHash,
		logNumHash: int(math.Log2(float64(numHashes))),
		indices:    indices,
		pos:        pos,
		permute:    permute,
	}
}

func (dw *DensifiedWtaHash) GetHash(indices []int, data []float64) []int {
	hashes := make([]int, dw.numHashes)
	hashArray := make([]int, dw.numHashes)
	values := make([]float64, dw.numHashes)

	for i := range hashes {
		hashes[i] = math.MinInt64
		values[i] = math.MinInt64
	}

	for p := 0; p < dw.permute; p++ {
		for i, dataValue := range data {
			binId := dw.indices[p*dw.rangePow+indices[i]]
			if binId < dw.numHashes && values[binId] < dataValue {
				values[binId] = dataValue
				hashes[binId] = dw.pos[p*dw.rangePow+indices[i]]
			}
		}
	}

	for i, next := range hashes {
		if next != math.MinInt64 {
			hashArray[i] = next
			continue
		}

		for count := 1; next == math.MinInt64; count++ {
			index := minInt(dw.GetRandDoubleHash(i, count), dw.numHashes-1)
			next = hashes[index] // Kills GPU.

			if count > 100 { // Densification failure.
				next = 0 // FIXME: can we do better than that?
				break
			}
		}

		hashArray[i] = next
	}

	return hashArray
}

func (dw *DensifiedWtaHash) GetHashEasy(data []float64, topK int) []int {
	hashes := make([]int, dw.numHashes)
	hashArray := make([]int, dw.numHashes)
	values := make([]float64, dw.numHashes)

	for i := range hashes {
		hashes[i] = math.MinInt64
		values[i] = math.MinInt64
	}

	for p := 0; p < dw.permute; p++ {
		binIndex := p * dw.rangePow
		for i, dataValue := range data {
			innerIndex := binIndex + i
			binId := dw.indices[innerIndex]
			if binId < dw.numHashes && values[binId] < dataValue {
				values[binId] = dataValue
				hashes[binId] = dw.pos[innerIndex]
			}
		}
	}

	for i, next := range hashes {
		if next != math.MinInt64 {
			hashArray[i] = next
			continue
		}

		for count := 1; next == math.MinInt64; count++ {
			index := minInt(dw.GetRandDoubleHash(i, count), dw.numHashes)
			next = hashes[index] // Kills GPU.

			if count > 100 { // Densification failure.
				break
			}
		}

		hashArray[i] = next
	}

	return hashArray
}

func (dw *DensifiedWtaHash) GetRandDoubleHash(binId, count int) int {
	toHash := ((uint(binId) + 1) << 6) + uint(count)
	return int((uint(dw.randHash[0]) * toHash << 3) >> (32 - dw.logNumHash)) // logNumHash needs to be ceiled.
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
