// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package densified_minihash

import "container/heap"

type indexValuePair struct {
	index int
	value float64
}

type indexValuePriorityQueue []indexValuePair

var _ heap.Interface = &indexValuePriorityQueue{}

func (p indexValuePriorityQueue) Len() int {
	return len(p)
}

func (p indexValuePriorityQueue) Less(i, j int) bool {
	// lower is better
	return p[i].value > p[j].value // TODO: check if correct
}

func (p indexValuePriorityQueue) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func (p *indexValuePriorityQueue) Push(x interface{}) {
	*p = append(*p, x.(indexValuePair))
}

func (p *indexValuePriorityQueue) Pop() (x interface{}) {
	n := len(*p)
	*p, x = (*p)[0:n-1], (*p)[n-1]
	return
}
