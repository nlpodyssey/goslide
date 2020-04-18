// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsh

import (
	"fmt"
	"testing"
)

func TestLSHNew(t *testing.T) {
	lsh := New(3, 4, 10)

	assertIntEqual(t, lsh.k, 3, "k")
	assertIntEqual(t, lsh.l, 4, "l")
	assertIntEqual(t, lsh.rangePow, 10, "rangePow")
	assertIntEqual(t, len(lsh.buckets), 4, "len(buckets)")
	assertIntEqual(t, len(lsh.rand1), 12, "len(rand1)")

	for _, b := range lsh.buckets {
		assertIntEqual(t, len(b), 0b10000000000, "len(buckets[])")
	}

	for i, r := range lsh.rand1 {
		isPositiveOdd(t, r, fmt.Sprintf("rand1[%d]", i))
	}
}

func TestLSHAdd(t *testing.T) {
	lsh := New(3, 4, 10)

	result := lsh.Add([]int{1, 10, 100, 1000}, 4321)
	assertIntSliceEqual(t, result, []int{0, 0, 0, 0}, "Add first result")

	result = lsh.Add([]int{2, 10, 200, 1000}, 4321)
	assertIntSliceEqual(t, result, []int{0, 1, 0, 1}, "Add second result")
}

func TestLSHAddSingle(t *testing.T) {
	lsh := New(3, 4, 10)

	result := lsh.AddSingle(0, 0, 123)
	assertIntEqual(t, result, 0, "Add first result")

	result = lsh.AddSingle(1, 1, 456)
	assertIntEqual(t, result, 0, "Add second result")

	result = lsh.AddSingle(0, 0, 789)
	assertIntEqual(t, result, 1, "Add third result")
}

func TestLSHRetrieveRaw(t *testing.T) {
	lsh := New(3, 4, 10)
	lsh.Add([]int{1, 10, 100, 1000}, 4321)
	result := lsh.RetrieveRaw([]int{1, 10, 100, 1000})

	assertIntEqual(t, len(result), 4, "len(RetrieveRaw)")

	for i, r := range result {
		assertIntEqual(t, len(r), 128, fmt.Sprintf("len(RetrieveRaw[%d])", i))

		assertIntEqual(t, r[0], 4321, fmt.Sprintf("RetrieveRaw[%d][0]", i))
		assertIntEqual(t, r[1], -1, fmt.Sprintf("RetrieveRaw[%d][1]", i))
	}
}

func TestLSHRetrieve(t *testing.T) {
	lsh := New(3, 4, 10)
	lsh.AddSingle(2, 3, 123)

	result := lsh.Retrieve(2, 3, 0)
	assertIntEqual(t, result, 123, "Retrieve")
}

func TestLSHHashesToIndex(t *testing.T) {
	// Just ensure no error is raised
	lsh := New(3, 4, 10)
	result := lsh.HashesToIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	assertIntEqual(t, len(result), 4, "len(HashesToIndex)")
}

func TestLSHClear(t *testing.T) {
	lsh := New(3, 4, 10)
	lsh.AddSingle(2, 3, 123)

	result := lsh.Retrieve(2, 3, 0)
	assertIntEqual(t, result, 123, "Retrieve before Clear")

	lsh.Clear()

	result = lsh.Retrieve(2, 3, 0)
	assertIntEqual(t, result, 0, "Retrieve after Clear")
}

func isPositiveOdd(t *testing.T, n int, msg string) {
	if n == 0 {
		t.Errorf("Assertion failed: %s | expected %d to be non zero", msg, n)
	} else if n%2 == 0 {
		t.Errorf("Assertion failed: %s | expected %d to odd", msg, n)
	}
}

func assertIntEqual(t *testing.T, actual, expected int, msg string) {
	if actual != expected {
		t.Errorf("Assertion failed: %s | expected %d, actual %d",
			msg, expected, actual)
	}
}

func assertIntSliceEqual(t *testing.T, actual, expected []int, msg string) {
	if len(actual) != len(expected) {
		t.Errorf("Lengths differ: %s | expected %d for %v, actual %d for %v",
			msg, len(expected), expected, len(actual), actual)
		return
	}

	for i, expVal := range expected {
		if actVal := actual[i]; expVal != actVal {
			t.Errorf("%s | values at %d differ: expected %d, actual %d",
				msg, i, expVal, actVal)
		}
	}
}
