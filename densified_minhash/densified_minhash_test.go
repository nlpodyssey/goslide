// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package densified_minhash

import "testing"

func TestDensifiedMinhashNew(t *testing.T) {
	h := New(3, 10)
	assertIntEqual(t, h.numHashes, 3, "numHashes")
	assertIntEqual(t, h.rangePow, 10, "rangePow")
	isPositiveOdd(t, h.randHash[0], "randHash[0]")
	isPositiveOdd(t, h.randHash[1], "randHash[1]")
	isPositiveOdd(t, h.randa, "randa")
	assertIntEqual(t, h.logNumHash, 1, "logNumHash")
}

func TestWtaHashGetHash(t *testing.T) {
	// Just ensure no error is raised
	h := New(3, 10)
	result := h.GetHash(
		[]int{0, 1, 2, 0, 1, 2, 0, 1, 2, 0},
		[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]int{0, 1, 2},
	)
	assertIntEqual(t, len(result), 3, "len(result)")
}

func TestWtaHashGetHashEasy(t *testing.T) {
	// Just ensure no error is raised
	h := New(3, 10)
	result := h.GetHashEasy(
		[]int{0, 1, 2, 0, 1, 2, 0, 1, 2, 0},
		[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		8,
	)
	assertIntEqual(t, len(result), 3, "len(result)")
}

func TestWtaHashGetRandDoubleHash(t *testing.T) {
	// Just ensure no error is raised
	h := New(3, 10)
	h.GetRandDoubleHash(1, 2)
}

func TestWtaHashGetMap(t *testing.T) {
	h := New(3, 10)
	result := h.GetMap(5)

	assertIntEqual(t, len(result), 5, "len(result)")

	for _, value := range result {
		if value < 0 || value >= 3 {
			t.Errorf("value expected to be in range 0-3, but got %d", value)
		}
	}
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
