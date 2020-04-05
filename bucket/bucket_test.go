// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bucket

import (
	"fmt"
	"testing"
)

func TestBucketNew(t *testing.T) {
	b := New()

	assertIntEqual(t, b.isInit, -1, "isInit")
	assertIntEqual(t, b.index, 0, "index")
	assertIntEqual(t, b.counts, 0, "counts")
	assertIntEqual(t, len(b.arr), 128, "len(arr)")

	assertIntEqual(t, b.GetSize(), 0, "GetSize")

	assertIntEqual(t, b.Retrieve(0), 0, "Retrieve(0)")
	assertIntEqual(t, b.Retrieve(127), 0, "Retrieve(127)")
	assertIntEqual(t, b.Retrieve(128), -1, "Retrieve(128)")

	all := b.GetAll()
	if all != nil {
		t.Errorf("Expected GetAll to return nil, but got %#v", all)
	}
}

func TestBucketAdd(t *testing.T) {
	b := New()

	t.Run("insert some initial elements", func(t *testing.T) {
		for i := 0; i < 10; i++ {
			b.Add(i + 1000)
		}

		assertIntEqual(t, b.GetSize(), 10, "GetSize")

		assertIntEqual(t, b.Retrieve(0), 1000, "Retrieve(0)")
		assertIntEqual(t, b.Retrieve(1), 1001, "Retrieve(1)")
		assertIntEqual(t, b.Retrieve(9), 1009, "Retrieve(9)")
		assertIntEqual(t, b.Retrieve(10), 0, "Retrieve(10)")
		assertIntEqual(t, b.Retrieve(127), 0, "Retrieve(127)")
		assertIntEqual(t, b.Retrieve(128), -1, "Retrieve(128)")

		all := b.GetAll()

		assertIntEqual(t, len(all), 128, "len(GetAll)")

		for i, v := range all {
			if i < 10 {
				assertIntEqual(t, v, i+1000, fmt.Sprintf("GetAll[%d]", i))
			} else if i == 10 {
				assertIntEqual(t, v, -1, fmt.Sprintf("GetAll[%d]", i))
			} else {
				assertIntEqual(t, v, 0, fmt.Sprintf("GetAll[%d]", i))
			}
		}
	})

	t.Run("insert more elements to reach full capacity", func(t *testing.T) {
		for i := 10; i < 128; i++ {
			b.Add(i + 1000)
		}

		assertIntEqual(t, b.GetSize(), 128, "GetSize")

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
			b.Add(i + 1000)
		}

		// FIXME: shouldn't size be still 128?
		assertIntEqual(t, b.GetSize(), 130, "GetSize")

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
