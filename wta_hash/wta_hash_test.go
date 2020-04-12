// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wta_hash

import "testing"

func TestWtaHashNew(t *testing.T) {
	h := New(3, 10)

	assertIntEqual(t, h.numHashes, 3, "numHashes")
	assertIntEqual(t, h.rangePow, 10, "rangePow")
	assertIntEqual(t, len(h.indices), 30, "indices len")

	for i := 0; i < 3; i++ {
		iOffset := i * 10

		var countByIndex [10]int

		for j := 0; j < 10; j++ {
			v := h.indices[iOffset+j]
			if v < 0 || v >= 10 {
				t.Errorf("Index %d out of range 0-9", v)
			}
			countByIndex[v]++
		}

		for index, count := range countByIndex {
			if count != 1 {
				t.Errorf(
					"Index %d expected to be present only once, but got %d",
					index, count)
			}
		}
	}
}

func TestWtaHashGetHash(t *testing.T) {
	h := New(3, 10)

	a := h.GetHash([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	b := h.GetHash([]float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100})
	assertIntSliceEqual(t, a, b, "a and b must be the same")

	c := h.GetHash([]float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1})
	assertIntSliceNotEqual(t, a, c, "a and c must differ")
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

func assertIntSliceNotEqual(t *testing.T, a, b []int, msg string) {
	if len(a) != len(b) {
		return
	}
	for i, aVal := range a {
		if bVal := b[i]; aVal != bVal {
			return
		}
	}
	t.Errorf("%s | values are expected to differ, but both are equal to %v",
		msg, a)
}
