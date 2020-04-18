// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fifo

const (
	bucketSize = 128
	bitMask    = bucketSize - 1
)

type FifoBucket struct {
	slice []int
	count int
}

func New() *FifoBucket {
	return &FifoBucket{
		slice: make([]int, 0, bucketSize),
		count: 0,
	}
}

func (b *FifoBucket) Reset() {
	b.slice = b.slice[:0]
	b.count = 0
}

func (b *FifoBucket) Add(id int) int {
	index := len(b.slice)
	if index == bucketSize {
		index = b.count & bitMask
		b.slice[index] = id
		b.count++
		return index
	}
	b.slice = append(b.slice, id)
	b.count++
	return index
}

func (b *FifoBucket) Retrieve(index int) int {
	if index >= bucketSize {
		return -1
	}
	if index >= len(b.slice) {
		return 0
	}
	return b.slice[index]
}

func (b *FifoBucket) GetAll() []int {
	return b.slice
}
