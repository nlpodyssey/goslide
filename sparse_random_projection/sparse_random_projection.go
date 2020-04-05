// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sparse_random_projection

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

type SparseRandomProjection struct {
	dim       int
	numHashes int
	samSize   int
	randBits  [][]bool
	indices   [][]int
}

func New(dimension, numOfHashes, ratio int) *SparseRandomProjection {
	samSize := int(math.Ceil(float64(dimension) / float64(ratio)))

	a := make([]int, dimension)
	for i := range a {
		a[i] = i
	}
	swap := func(i, j int) { a[i], a[j] = a[j], a[i] }

	rand.Seed(time.Now().UnixNano())
	randBits := make([][]bool, numOfHashes)
	indices := make([][]int, numOfHashes)

	for i := 0; i < numOfHashes; i++ {
		rand.Shuffle(dimension, swap)

		randBits[i] = make([]bool, samSize)
		indices[i] = make([]int, samSize)

		for j := 0; j < samSize; j++ {
			indices[i][j] = a[j]
			randBits[i][j] = rand.Int()%2 == 0
		}

		sort.Ints(indices[i])
	}

	return &SparseRandomProjection{
		dim:       dimension,
		numHashes: numOfHashes,
		samSize:   samSize,
		randBits:  randBits,
		indices:   indices,
	}
}

func (srp *SparseRandomProjection) GetHash(vector []float64) []int {
	hashes := make([]int, srp.numHashes)

	// TODO: parallel?
	for i := range hashes {
		s := 0.0
		for j, rb := range srp.randBits[i] {
			v := vector[srp.indices[i][j]]
			if rb {
				s += v
			} else {
				s -= v
			}
		}

		if s >= 0 {
			hashes[i] = 0
		} else {
			hashes[i] = 1
		}
	}

	return hashes
}

func (srp *SparseRandomProjection) GetHashSparse(
	indices []int,
	values []float64,
) []int {
	length := len(indices)
	hashes := make([]int, srp.numHashes)

	for p := range hashes {
		s := 0.0

		for i, j := 0, 0; i < length && j < srp.samSize; {
			if indices[i] == srp.indices[p][j] {
				v := values[i]
				if srp.randBits[p][j] {
					s += v
				} else {
					s -= v
				}
				i++
				j++
			} else if indices[i] < srp.indices[p][j] {
				i++
			} else {
				j++
			}
		}

		if s >= 0 {
			hashes[p] = 0
		} else {
			hashes[p] = 1
		}
	}

	return hashes
}
