// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package node

import (
	"github.com/nlpodyssey/goslide/configuration"
)

type Node struct {
	cowId int // "thread" ID for copy on write
	base  *baseNode
	train []*NodeTrain
}

type baseNode struct {
	cowId            int // "thread" ID for copy on write
	activeInputs     int
	nodeType         NodeType
	currentBatchsize int
	dim              int
	layerNum         int
	idInLayer        int
	indicesInTables  []int
	indicesInBuckets []int
	weights          []float64
	mirrorWeights    []float64
	adamAvgMom       []float64
	adamAvgVel       []float64
	t                []float64 // for adam
	update           []int
	bias             float64
	tBias            float64
	adamAvgMomBias   float64
	adamAvgVelBias   float64
	mirrorBias       float64
}

type NodeTrain struct {
	cowId           int // "thread" ID for copy on write
	lastDeltaforBPs float64
	lastActivations float64
	lastGradients   float64
	activeinputIds  float64
}

func NewNodeTrain(cowId int) *NodeTrain {
	return &NodeTrain{
		cowId: cowId,
	}
}

func NewEmptyNode(cowId int) *Node {
	return &Node{
		cowId: cowId,
		base:  &baseNode{cowId: cowId},
		train: nil,
	}
}

func NewNode(
	cowId int,
	dim int,
	nodeId int,
	layerId int,
	nodeType NodeType,
	batchsize int,
	weights []float64,
	bias float64,
	adamAvgMom []float64,
	adamAvgVel []float64,
) *Node {
	newNode := &Node{
		cowId: cowId,
		base: &baseNode{
			cowId:            cowId,
			dim:              dim,
			idInLayer:        nodeId,
			nodeType:         nodeType,
			layerNum:         layerId,
			currentBatchsize: batchsize,
			activeInputs:     0,
			weights:          weights,
			bias:             bias,
			mirrorBias:       bias,
		},
		train: make([]*NodeTrain, batchsize),
	}

	if configuration.Global.UseAdam {
		newNode.base.adamAvgMom = adamAvgMom
		newNode.base.adamAvgVel = adamAvgVel
		newNode.base.t = make([]float64, dim)
	}

	for i := range newNode.train {
		newNode.train[i] = NewNodeTrain(cowId)
	}

	return newNode
}

func (n *Node) Weights() []float64 {
	return n.base.weights
}

// Copies the given weights into the node (without cloning)
func (n *Node) SetWeights(weights []float64) {
	copy(n.base.weights, weights)
}

func (n *Node) Bias() float64 {
	return n.base.bias
}

func (n *Node) Dim() int {
	return n.base.dim
}

func (n *Node) GetT(i int) float64 {
	return n.base.t[i]
}

func (nd *Node) SetT(cowId, i int, value float64) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)
	n.base.t[i] = value
	return n
}

func (n *Node) GetAdamAvgMom(i int) float64 {
	return n.base.adamAvgMom[i]
}

func (nd *Node) SetAdamAvgMom(cowId, i int, value float64) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)
	n.base.adamAvgMom[i] = value
	return n
}

func (n *Node) GetAdamAvgVel(i int) float64 {
	return n.base.adamAvgVel[i]
}

func (nd *Node) SetAdamAvgVel(cowId, i int, value float64) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)
	n.base.adamAvgVel[i] = value
	return n
}

func (n *Node) GetAdamAvgMomBias() float64 {
	return n.base.adamAvgMomBias
}

func (nd *Node) SetAdamAvgMomBias(cowId int, value float64) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)
	n.base.adamAvgMomBias = value
	return n
}

func (n *Node) GetAdamAvgVelBias() float64 {
	return n.base.adamAvgVelBias
}

func (nd *Node) SetAdamAvgVelBias(cowId int, value float64) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)
	n.base.adamAvgVelBias = value
	return n
}

func (n *Node) GetBias() float64 {
	return n.base.bias
}

func (nd *Node) SetBias(cowId int, value float64) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)
	n.base.bias = value
	return n
}

func (n *Node) GetTBias() float64 {
	return n.base.tBias
}

func (nd *Node) SetTBias(cowId int, value float64) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)
	n.base.tBias = value
	return n
}

func (nd *Node) CopyWeightsAndBiasFromMirror(cowId int) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)
	copy(n.base.weights, n.base.mirrorWeights)
	n.base.bias = n.base.mirrorBias
	return n
}

func (nd *Node) SetIndices(
	cowId int,
	indicesInTables []int,
	indicesInBuckets []int,
) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)

	n.base.indicesInTables = indicesInTables
	n.base.indicesInBuckets = indicesInBuckets

	return n
}

func (nd *Node) Update(
	cowId int,
	dim int,
	nodeId int,
	layerId int,
	nodeType NodeType,
	batchsize int,
	weights []float64,
	bias float64,
	adamAvgMom []float64,
	adamAvgVel []float64,
	trainBlob []*NodeTrain,
) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)

	n.base.dim = dim
	n.base.idInLayer = nodeId
	n.base.nodeType = nodeType
	n.base.layerNum = layerId
	n.base.currentBatchsize = batchsize
	n.base.activeInputs = 0
	n.base.weights = weights
	n.base.bias = bias
	n.base.mirrorBias = bias

	if configuration.Global.UseAdam {
		n.base.adamAvgMom = adamAvgMom
		n.base.adamAvgVel = adamAvgVel
		n.base.t = make([]float64, dim)
	}

	n.train = trainBlob

	return n
}

func (n *Node) GetLastActivation(inputId int) float64 {
	t := n.train[inputId]
	if t.activeinputIds != 1 {
		return 0
	}
	return t.lastActivations
}

func (n *Node) GetInputActive(inputId int) bool {
	return n.train[inputId].activeinputIds == 1
}

func (n *Node) GetActiveInputs() bool {
	return n.base.activeInputs > 0
}

func (nd *Node) IncrementDelta(
	cowId, inputId int,
	incrementValue float64,
) *Node {
	if nd.train[inputId].activeinputIds != 1 {
		panic("Input Not Active but still called")
	} else if nd.train[inputId].lastActivations <= 0 {
		return nd
	}

	n := nd.cloneIfNeeded(cowId)
	n.train[inputId] = n.train[inputId].cloneIfNeeded(cowId)
	n.train[inputId].lastDeltaforBPs += incrementValue
	return n
}

func (nd *Node) GetActivation(
	cowId int,
	indices []int,
	values []float64,
	length int,
	inputId int,
) (float64, *Node) {
	if inputId > nd.base.currentBatchsize {
		panic("Input ID more than Batch Size")
	}

	//FUTURE TODO: shrink batchsize and check if input is already active then ignore and ensure backpopagation is ignored too.

	n := nd.cloneIfNeeded(cowId)
	n.train[inputId] = n.train[inputId].cloneIfNeeded(cowId)

	if n.train[inputId].activeinputIds != 1 {
		n.train[inputId].activeinputIds = 1 // activate input

		n.base = n.base.cloneIfNeeded(cowId)
		n.base.activeInputs++
	}

	n.train[inputId].lastActivations = 0

	for i := 0; i < length; i++ {
		n.train[inputId].lastActivations +=
			n.base.weights[indices[i]] * values[i]
	}

	n.train[inputId].lastActivations += n.base.bias

	switch n.base.nodeType {
	case ReLU:
		if n.train[inputId].lastActivations < 0 {
			n.train[inputId].lastActivations = 0
			n.train[inputId].lastGradients = 1
			n.train[inputId].lastDeltaforBPs = 0
		} else {
			n.train[inputId].lastGradients = 0
		}
	case Softmax: // do nothing
	default:
		panic("Invalid Node type from Constructor")
	}

	return n.train[inputId].lastActivations, n
}

func (nd *Node) ComputeExtaStatsForSoftMax(
	cowId int,
	normalizationConstant float64,
	inputId int,
	labels []int,
) *Node {
	if nd.train[inputId].activeinputIds != 1 {
		panic("Input Not Active but still called")
	}

	n := nd.cloneIfNeeded(cowId)
	n.train[inputId] = n.train[inputId].cloneIfNeeded(cowId)

	n.train[inputId].lastActivations /= normalizationConstant + 0.0000001

	//TODO:check gradient
	n.train[inputId].lastGradients = 1

	if intSliceContains(labels, n.base.idInLayer) {
		n.train[inputId].lastDeltaforBPs = 1.0/float64(len(labels)) -
			n.train[inputId].lastActivations/float64(n.base.currentBatchsize)
	} else {
		n.train[inputId].lastDeltaforBPs = -n.train[inputId].lastActivations /
			float64(n.base.currentBatchsize)
	}

	return n
}

func (nd *Node) BackPropagate(
	cowId int,
	previousNodes *[]*Node,
	previousLayerActiveNodeIds []int,
	learningRate float64,
	inputId int,
) *Node {
	if nd.train[inputId].activeinputIds != 1 {
		panic("Input Not Active but still called")
	}

	n := nd.cloneIfNeeded(cowId)
	n.train[inputId] = n.train[inputId].cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)

	for _, nodeId := range previousLayerActiveNodeIds {
		// Update Delta before updating weights
		prevNode := (*previousNodes)[nodeId].IncrementDelta(
			cowId,
			inputId,
			n.train[inputId].lastDeltaforBPs*n.base.weights[nodeId])
		(*previousNodes)[nodeId] = prevNode

		gradT := n.train[inputId].lastDeltaforBPs *
			prevNode.GetLastActivation(inputId)

		if configuration.Global.UseAdam {
			n.base.t[nodeId] += gradT
		} else {
			n.base.mirrorWeights[nodeId] += learningRate * gradT
		}
	}

	if configuration.Global.UseAdam {
		biasgradT := n.train[inputId].lastDeltaforBPs
		// TODO: ?? biasgradTsq := biasgradT * biasgradT
		n.base.tBias += biasgradT
	} else {
		n.base.mirrorBias += learningRate * n.train[inputId].lastDeltaforBPs
	}

	n.train[inputId].activeinputIds = 0
	n.train[inputId].lastDeltaforBPs = 0
	n.train[inputId].lastActivations = 0
	n.base.activeInputs--

	return n
}

func (nd *Node) BackPropagateFirstLayer(
	cowId int,
	nnzIndices []int,
	nnzValues []float64,
	nnzSize int,
	learningRate float64,
	inputId int,
) *Node {
	if nd.train[inputId].activeinputIds != 1 {
		panic("Input Not Active but still called")
	}

	n := nd.cloneIfNeeded(cowId)
	n.train[inputId] = n.train[inputId].cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)

	for i := 0; i < nnzSize; i++ {
		gradT := n.train[inputId].lastDeltaforBPs * nnzValues[i]
		// TODO: ?? gradTsq := gradT * gradT
		if configuration.Global.UseAdam {
			n.base.t[nnzIndices[i]] += gradT
		} else {
			n.base.mirrorWeights[nnzIndices[i]] += learningRate * gradT
		}
	}

	if configuration.Global.UseAdam {
		biasgradT := n.train[inputId].lastDeltaforBPs
		// TODO: ?? biasgradTsq = biasgradT * biasgradT
		n.base.tBias += biasgradT
	} else {
		n.base.mirrorBias += learningRate + n.train[inputId].lastDeltaforBPs
	}

	n.train[inputId].activeinputIds = 0 // deactivate inputIDs
	n.train[inputId].lastDeltaforBPs = 0
	n.train[inputId].lastActivations = 0
	n.base.activeInputs--

	return n
}

func (nd *Node) SetlastActivation(
	cowId int,
	inputId int,
	realActivation float64,
) *Node {
	n := nd.cloneIfNeeded(cowId)
	n.train[inputId] = n.train[inputId].cloneIfNeeded(cowId)

	n.train[inputId].lastActivations = realActivation

	return n
}

// for debugging gradients.
func (nd *Node) PurturbWeight(
	cowId int,
	weightId int,
	delta float64,
) (float64, *Node) {
	n := nd.cloneIfNeeded(cowId)
	n.base = n.base.cloneIfNeeded(cowId)

	n.base.weights[weightId] += delta

	return n.base.weights[weightId], n
}

func (n *Node) GetGradient(
	cowId int,
	weightId, inputId int,
	inputVal float64,
) float64 {
	return -n.train[inputId].lastDeltaforBPs * inputVal
}

func (n *Node) cloneIfNeeded(cowId int) *Node {
	if n.cowId != cowId {
		return n.clone(cowId)
	}
	return n
}

func (n *Node) clone(cowId int) *Node {
	newNode := &Node{
		cowId: cowId,
		base:  n.base,
		train: nil,
	}

	if n.train != nil {
		newNode.train = make([]*NodeTrain, len(n.train))
		copy(newNode.train, n.train)
	}

	return newNode
}

func (n *baseNode) cloneIfNeeded(cowId int) *baseNode {
	if n.cowId != cowId {
		return n.clone(cowId)
	}
	return n
}

func (n *baseNode) clone(cowId int) *baseNode {
	// TODO: not sure we should clone all slices...
	return &baseNode{
		cowId:            cowId,
		activeInputs:     n.activeInputs,
		nodeType:         n.nodeType,
		currentBatchsize: n.currentBatchsize,
		dim:              n.dim,
		layerNum:         n.layerNum,
		idInLayer:        n.idInLayer,
		indicesInTables:  copyIntSlice(n.indicesInTables),
		indicesInBuckets: copyIntSlice(n.indicesInBuckets),
		weights:          copyFloat64Slice(n.weights),
		mirrorWeights:    copyFloat64Slice(n.mirrorWeights),
		adamAvgMom:       copyFloat64Slice(n.adamAvgMom),
		adamAvgVel:       copyFloat64Slice(n.adamAvgVel),
		t:                copyFloat64Slice(n.t),
		update:           copyIntSlice(n.update),
		bias:             n.bias,
		tBias:            n.tBias,
		adamAvgMomBias:   n.adamAvgMomBias,
		adamAvgVelBias:   n.adamAvgVelBias,
		mirrorBias:       n.mirrorBias,
	}
}

func (n *NodeTrain) cloneIfNeeded(cowId int) *NodeTrain {
	if n.cowId != cowId {
		return n.clone(cowId)
	}
	return n
}

func (t *NodeTrain) clone(cowId int) *NodeTrain {
	return &NodeTrain{
		cowId:           cowId,
		lastDeltaforBPs: t.lastDeltaforBPs,
		lastActivations: t.lastActivations,
		lastGradients:   t.lastGradients,
		activeinputIds:  t.activeinputIds,
	}
}

func copyIntSlice(s []int) []int {
	newS := make([]int, len(s))
	copy(newS, s)
	return s
}

func copyFloat64Slice(s []float64) []float64 {
	newS := make([]float64, len(s))
	copy(newS, s)
	return s
}

func intSliceContains(slice []int, value int) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}
