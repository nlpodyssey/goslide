// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsh

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/nlpodyssey/goslide/bucket"
	"github.com/nlpodyssey/goslide/configuration"
)

const binSize = 8

var logBinSize = int(math.Floor(math.Log(float64(binSize))))

type LSH struct {
	buckets  [][]*bucket.Bucket
	k        int
	l        int
	rangePow int
	rand1    []int
}

func New(k, l, rangePow int) *LSH {
	buckets := make([][]*bucket.Bucket, l)

	// TODO: parallel?
	for i := range buckets {
		buckets[i] = make([]*bucket.Bucket, 1<<rangePow)
		for j := range buckets[i] {
			buckets[i][j] = bucket.New()
		}
	}

	rand1 := make([]int, k*l)

	// TODO: parallel?
	for i := range rand1 {
		rand1[i] = positiveOddRandomInt()
	}

	return &LSH{
		buckets:  buckets,
		k:        k,
		l:        l,
		rangePow: rangePow,
		rand1:    rand1,
	}
}

func (lsh *LSH) Clear() {
	for _, buckets := range lsh.buckets {
		for _, bucket := range buckets {
			bucket.Reset()
		}
	}
}

func (lsh *LSH) Count() {
	for i, bi := range lsh.buckets {
		total := 0
		for _, bj := range bi {
			size := bj.GetSize()
			if size != 0 {
				fmt.Printf("%d ", size)
			}
			total += size
		}
		fmt.Printf("\nTABLE %d Total %d\n", i, total)
	}
}

func (lsh *LSH) HashesToIndex(hashes []int) []int {
	hashFunction := configuration.Global.HashFunction

	indices := make([]int, lsh.l)

	for i := range indices {
		var index uint = 0

		for j := 0; j < lsh.k; j++ {
			switch hashFunction {
			case configuration.WtaHashFunction, configuration.DensifiedWtaHashFunction:
				h := uint(hashes[lsh.k*i+j])
				index += h << ((lsh.k - 1 - j) * logBinSize)
			case configuration.DensifiedMinhashFunction:
				randVal := uint(lsh.rand1[lsh.k*i+j])
				h := randVal * randVal
				h ^= h >> 13
				h ^= uint(lsh.rand1[lsh.k*i+j])
				index += h * uint(hashes[lsh.k*i+j])
			case configuration.SparseRandomProjectionHashFunction:
				h := uint(hashes[lsh.k*i+j])
				index += h << (lsh.k - 1 - j)
			}
		}

		if hashFunction == configuration.DensifiedMinhashFunction {
			index &= ((1 << lsh.rangePow) - 1)
		}

		indices[i] = int(index)
	}

	return indices
}

func (lsh *LSH) Add(indices []int, id int) []int {
	secondIndices := make([]int, lsh.l)
	for i := range secondIndices {
		secondIndices[i] = lsh.buckets[i][indices[i]].Add(id)
	}
	return secondIndices
}

func (lsh *LSH) AddSingle(tableId, index, id int) int {
	return lsh.buckets[tableId][index].Add(id)
}

// Returns all the buckets
func (lsh *LSH) RetrieveRaw(indices []int) [][]int {
	rawResults := make([][]int, lsh.l)
	for i := range rawResults {
		rawResults[i] = lsh.buckets[i][indices[i]].GetAll()
	}
	return rawResults
}

func (lsh *LSH) Retrieve(table, index, bucket int) int {
	return lsh.buckets[table][index].Retrieve(bucket)
}

func positiveOddRandomInt() int {
	n := rand.Int()
	if n%2 == 0 {
		return n + 1
	}
	return n
}
