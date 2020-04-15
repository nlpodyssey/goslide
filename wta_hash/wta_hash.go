// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of Winner Take All (WTA) hash
//
// Algorithm from the paper:
//   The Power of Comparative Reasoning
//   Jay Yagnik, Dennis Strelow, David A. Ross, Ruei-sung Lin
//   https://www.cs.toronto.edu/~dross/YagnikStrelowRossLin_ICCV2011.pdf
package wta_hash

import (
	"math"
	"math/rand"
	"time"

	"github.com/nlpodyssey/goslide/index_value"
)

// The number of times the range is larger than
// the total number of hashes we need.
const binSize = 8

type WtaHash struct {
	indices   []int
	numHashes int
	rangePow  int
}

func New(numHashes, numOfBitsToHash int) *WtaHash {
	rand.Seed(time.Now().UnixNano())

	permute := int(math.Ceil(
		float64(numHashes) * float64(binSize) / float64(numOfBitsToHash)))
	nArray := make([]int, numOfBitsToHash)
	indices := make([]int, numOfBitsToHash*permute)

	for i := range nArray {
		nArray[i] = i
	}

	swap := func(i, j int) {
		nArray[i], nArray[j] = nArray[j], nArray[i]
	}

	for p := 0; p < permute; p++ {
		rand.Shuffle(numOfBitsToHash, swap)

		firstIndex := p * numOfBitsToHash
		lastIndex := firstIndex + numOfBitsToHash
		copy(indices[firstIndex:lastIndex], nArray)
	}

	return &WtaHash{
		indices:   indices,
		numHashes: numHashes,
		rangePow:  numOfBitsToHash,
	}
}

func (wh *WtaHash) GetHash(data []index_value.Pair) []int {
	hashes := make([]int, wh.numHashes)
	values := make([]float64, wh.numHashes)

	for i := 0; i < wh.numHashes; i++ {
		hashes[i] = math.MinInt64
		values[i] = math.MinInt64
	}

	for i := 0; i < wh.numHashes; i++ {
		iOffset := i * binSize

		for j := 0; j < binSize; j++ {
			jOffset := iOffset + j

			curIndex := wh.indices[jOffset]
			if value := data[curIndex].Value; values[i] < value {
				hashes[i] = curIndex
				values[i] = value
			}
		}
	}

	return hashes
}

// TODO: avoid code duplication
func (wh *WtaHash) GetHashDense(data []float64) []int {
	hashes := make([]int, wh.numHashes)
	values := make([]float64, wh.numHashes)

	for i := 0; i < wh.numHashes; i++ {
		hashes[i] = math.MinInt64
		values[i] = math.MinInt64
	}

	for i := 0; i < wh.numHashes; i++ {
		iOffset := i * binSize

		for j := 0; j < binSize; j++ {
			jOffset := iOffset + j

			curIndex := wh.indices[jOffset]
			if value := data[curIndex]; values[i] < value {
				hashes[i] = curIndex
				values[i] = value
			}
		}
	}

	return hashes
}
