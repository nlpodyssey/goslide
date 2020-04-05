// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sparse_random_projection

import "testing"

func TestSparseRandomProjectionNew(t *testing.T) {
	h := New(10, 3, 2)
	assertIntEqual(t, h.dim, 10, "dim")
	assertIntEqual(t, h.numHashes, 3, "numHashes")
	assertIntEqual(t, h.samSize, 5, "samSize")

	assertIntEqual(t, len(h.randBits), 3, "len(randBits)")
	assertIntEqual(t, len(h.indices), 3, "len(indices)")

	for _, value := range h.randBits {
		assertIntEqual(t, len(value), 5, "len(randBits[])")
	}

	for _, value := range h.indices {
		assertIntEqual(t, len(value), 5, "len(indices[])")
		for _, innerValue := range value {
			if innerValue < 0 || innerValue >= 10 { // depends on binSize
				t.Errorf("value expected to be in range 0-9, but got %d", innerValue)
			}
		}
	}
}

func TestSparseRandomProjectionGetHash(t *testing.T) {
	// Just ensure no error is raised
	h := New(10, 3, 2)
	result := h.GetHash([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	assertIntEqual(t, len(result), 3, "len(result)")
}

func TestSparseRandomProjectionGetHashSparse(t *testing.T) {
	// Just ensure no error is raised
	h := New(10, 3, 2)
	result := h.GetHashSparse(
		[]int{0, 4, 7, 9},
		[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	)
	assertIntEqual(t, len(result), 3, "len(result)")
}

func assertIntEqual(t *testing.T, actual, expected int, msg string) {
	if actual != expected {
		t.Errorf("Assertion failed: %s | expected %d, actual %d",
			msg, expected, actual)
	}
}
