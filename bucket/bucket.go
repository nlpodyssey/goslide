// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bucket

import "math/rand"

const bucketSize = 128
const fifo = true

type Bucket struct {
	arr    []int
	isInit int
	index  int
	counts int
}

func New() *Bucket {
	return &Bucket{
		arr:    make([]int, bucketSize),
		isInit: -1,
		index:  0,
		counts: 0,
	}
}

func (b *Bucket) GetSize() int {
	return b.counts
}

func (b *Bucket) Add(id int) int {
	if fifo { // FIFO
		b.isInit++
		index := b.counts & (bucketSize - 1)
		b.counts++
		b.arr[index] = id
		return index
	} else { // Reservoir Sampling
		b.counts++
		if b.index == bucketSize {
			randNum := rand.Intn(b.counts) + 1
			if randNum == 2 {
				randIndex := rand.Intn(bucketSize)
				b.arr[randIndex] = id
				return randIndex
			} else {
				return -1
			}
		} else {
			b.arr[b.index] = id
			returnIndex := b.index
			b.index++
			return returnIndex
		}
	}
}

func (b *Bucket) Retrieve(index int) int {
	if index >= bucketSize {
		return -1
	}
	return b.arr[index]
}

func (b *Bucket) GetAll() []int {
	if b.isInit == -1 {
		return nil
	}
	if b.counts < bucketSize {
		b.arr[b.counts] = -1
	}
	return b.arr
}
