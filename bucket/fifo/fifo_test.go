// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fifo

import (
	"fmt"
	"testing"
)

func TestFifoBucketNew(t *testing.T) {
	b := New()

	assertIntEqual(t, b.count, 0, "counts")
	assertIntEqual(t, len(b.slice), 0, "len(slice)")
	assertIntEqual(t, cap(b.slice), 128, "cap(slice)")

	assertIntEqual(t, b.Retrieve(0), 0, "Retrieve(0)")
	assertIntEqual(t, b.Retrieve(127), 0, "Retrieve(127)")
	assertIntEqual(t, b.Retrieve(128), -1, "Retrieve(128)")

	all := b.GetAll()
	assertIntEqual(t, len(all), 0, "len(GetAll)")
}

func TestFifoBucketAdd(t *testing.T) {
	b := New()

	t.Run("insert some initial elements", func(t *testing.T) {
		for i := 0; i < 10; i++ {
			r := b.Add(i + 1000)
			assertIntEqual(t, r, i, "Add")
		}

		assertIntEqual(t, b.count, 10, "counts")
		assertIntEqual(t, len(b.slice), 10, "len(slice)")
		assertIntEqual(t, cap(b.slice), 128, "cap(slice)")

		assertIntEqual(t, b.Retrieve(0), 1000, "Retrieve(0)")
		assertIntEqual(t, b.Retrieve(1), 1001, "Retrieve(1)")
		assertIntEqual(t, b.Retrieve(9), 1009, "Retrieve(9)")
		assertIntEqual(t, b.Retrieve(10), 0, "Retrieve(10)")
		assertIntEqual(t, b.Retrieve(127), 0, "Retrieve(127)")
		assertIntEqual(t, b.Retrieve(128), -1, "Retrieve(128)")

		all := b.GetAll()

		assertIntEqual(t, len(all), 10, "len(GetAll)")

		for i, v := range all {
			assertIntEqual(t, v, i+1000, fmt.Sprintf("GetAll[%d]", i))
		}
	})

	t.Run("insert more elements to reach full capacity", func(t *testing.T) {
		for i := 10; i < 128; i++ {
			r := b.Add(i + 1000)
			assertIntEqual(t, r, i, "Add")
		}

		assertIntEqual(t, b.count, 128, "counts")
		assertIntEqual(t, len(b.slice), 128, "len(slice)")
		assertIntEqual(t, cap(b.slice), 128, "cap(slice)")

		assertIntEqual(t, b.Retrieve(0), 1000, "Retrieve(0)")
		assertIntEqual(t, b.Retrieve(1), 1001, "Retrieve(1)")
		assertIntEqual(t, b.Retrieve(9), 1009, "Retrieve(9)")
		assertIntEqual(t, b.Retrieve(10), 1010, "Retrieve(10)")
		assertIntEqual(t, b.Retrieve(127), 1127, "Retrieve(127)")
		assertIntEqual(t, b.Retrieve(128), -1, "Retrieve(128)")

		all := b.GetAll()

		for i, v := range all {
			assertIntEqual(t, v, i+1000, fmt.Sprintf("GetAll[%d]", i))
		}
	})

	t.Run("insert more elements beyond capacity", func(t *testing.T) {
		for i := 128; i < 130; i++ {
			r := b.Add(i + 1000)
			assertIntEqual(t, r, i-128, "Add")
		}

		assertIntEqual(t, b.count, 130, "counts")
		assertIntEqual(t, len(b.slice), 128, "len(slice)")
		assertIntEqual(t, cap(b.slice), 128, "cap(slice)")

		assertIntEqual(t, b.Retrieve(0), 1128, "Retrieve(0)")
		assertIntEqual(t, b.Retrieve(1), 1129, "Retrieve(1)")
		assertIntEqual(t, b.Retrieve(2), 1002, "Retrieve(2)")
		assertIntEqual(t, b.Retrieve(3), 1003, "Retrieve(3)")
		assertIntEqual(t, b.Retrieve(127), 1127, "Retrieve(127)")
		assertIntEqual(t, b.Retrieve(128), -1, "Retrieve(128)")

		all := b.GetAll()

		for i, v := range all {
			if i < 2 {
				assertIntEqual(t, v, i+1128, fmt.Sprintf("GetAll[%d]", i))
			} else {
				assertIntEqual(t, v, i+1000, fmt.Sprintf("GetAll[%d]", i))
			}
		}
	})
}

func assertIntEqual(t *testing.T, actual, expected int, msg string) {
	if actual != expected {
		t.Errorf("Assertion failed: %s | expected %d, actual %d",
			msg, expected, actual)
	}
}
