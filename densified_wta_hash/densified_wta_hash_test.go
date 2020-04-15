// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package densified_wta_hash

import (
	"testing"

	"github.com/nlpodyssey/goslide/index_value"
)

func TestDensifiedWtaHashNew(t *testing.T) {
	h := New(3, 10)
	assertIntEqual(t, h.numHashes, 3, "numHashes")
	assertIntEqual(t, h.rangePow, 10, "rangePow")
	isPositiveOdd(t, h.randHash[0], "randHash[0]")
	isPositiveOdd(t, h.randHash[1], "randHash[1]")
	isPositiveOdd(t, h.randa, "randa")
	assertIntEqual(t, h.logNumHash, 1, "logNumHash")
	assertIntEqual(t, h.permute, 3, "permute")
	assertIntEqual(t, len(h.indices), 30, "len(indices)")
	assertIntEqual(t, len(h.pos), 30, "len(pos)")

	for _, value := range h.indices {
		if value < 0 || value >= 4 { // depends on binSize
			t.Errorf("value expected to be in range 0-3, but got %d", value)
		}
	}

	for _, value := range h.pos {
		if value < 0 || value >= 8 { // 8 is binSize
			t.Errorf("value expected to be in range 0-8, but got %d", value)
		}
	}
}

func TestDensifiedWtaHashGetHash(t *testing.T) {
	// Just ensure no error is raised
	h := New(3, 10)
	result := h.GetHash(
		[]index_value.Pair{
			{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
			{5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10},
		},
	)
	assertIntEqual(t, len(result), 3, "len(result)")
}

func TestDensifiedWtaHashGetHashEasy(t *testing.T) {
	// Just ensure no error is raised
	h := New(3, 10)
	result := h.GetHashEasy(
		[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		8,
	)
	assertIntEqual(t, len(result), 3, "len(result)")
}

func TestDensifiedWtaHashGetRandDoubleHash(t *testing.T) {
	// Just ensure no error is raised
	h := New(3, 10)
	h.GetRandDoubleHash(1, 2)
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
