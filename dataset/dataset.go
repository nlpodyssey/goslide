// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset

import "github.com/nlpodyssey/goslide/index_value"

type Example struct {
	Features []index_value.Pair
	Labels   []int
}
