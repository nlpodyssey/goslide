// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package node

type NodeType int8

const (
	ReLU NodeType = iota
	Softmax
)

var nodeTypeStringValues = [...]string{
	"ReLU",
	"Softmax",
}

func (nt NodeType) String() string {
	return nodeTypeStringValues[nt]
}
