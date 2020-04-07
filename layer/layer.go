// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layer

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/nlpodyssey/goslide/configuration"
	"github.com/nlpodyssey/goslide/densified_minhash"
	"github.com/nlpodyssey/goslide/densified_wta_hash"
	"github.com/nlpodyssey/goslide/lsh"
	"github.com/nlpodyssey/goslide/node"
	"github.com/nlpodyssey/goslide/sparse_random_projection"
	"github.com/nlpodyssey/goslide/wta_hash"
)

const srpRatio = 32
const topK = 30
const normalDistributionStdDev = 0.01
const normalDistributionMean = 0.0

type Layer struct {
	cowId                   int // "thread" ID for copy on write
	nodeType                node.NodeType
	nodes                   []*node.Node
	randNode                []int
	normalizationConstants  []float64
	k                       int
	l                       int
	rangeRow                int
	previousLayerNumOfNodes int
	batchSize               int
	trainArray              []*node.NodeTrain
	layerId                 int
	numOfActive             int
	numOfNodes              int
	weights                 []float64
	adamAvgMom              []float64
	adamAvgVel              []float64
	bias                    []float64
	hashTables              *lsh.LSH
	wtaHasher               *wta_hash.WtaHash
	minHasher               *densified_minhash.DensifiedMinhash
	srp                     *sparse_random_projection.SparseRandomProjection
	dwtaHasher              *densified_wta_hash.DensifiedWtaHash
	binIds                  []int
}

func New(
	cowId int,
	numOfNodes int,
	previousLayerNumOfNodes int,
	layerId int,
	nodeType node.NodeType,
	batchSize int,
	k int,
	l int,
	rangePow int,
	sparsity float64,
	weights []float64,
	bias []float64,
	adamAvgMom []float64,
	adamAvgVel []float64,
) *Layer {
	nodes := make([]*node.Node, numOfNodes)
	for i := range nodes {
		nodes[i] = node.NewEmptyNode(cowId)
	}

	// Create a list of random nodes just in case not enough nodes
	// from hashtable for active nodes.
	randNode := make([]int, numOfNodes)
	for i := range randNode {
		randNode[i] = i
	}
	swapRandNode := func(i, j int) {
		randNode[i], randNode[j] = randNode[j], randNode[i]
	}
	rand.Shuffle(numOfNodes, swapRandNode)

	newLayer := &Layer{
		cowId:                   cowId,
		nodeType:                nodeType,
		nodes:                   nodes,
		randNode:                randNode,
		normalizationConstants:  nil,
		k:                       k,
		l:                       l,
		rangeRow:                rangePow,
		previousLayerNumOfNodes: previousLayerNumOfNodes,
		batchSize:               batchSize,
		trainArray:              nil,
		layerId:                 layerId,
		numOfActive:             int(math.Floor(float64(numOfNodes) * sparsity)),
		numOfNodes:              numOfNodes,
		weights:                 nil,
		adamAvgMom:              nil,
		adamAvgVel:              nil,
		bias:                    nil,
		// TODO: Initialize Hash Tables and add the nodes.
		hashTables: lsh.New(k, l, rangePow),
		wtaHasher:  nil,
		minHasher:  nil,
		srp:        nil,
		dwtaHasher: nil,
		binIds:     nil,
	}

	switch configuration.Global.HashFunction {
	case configuration.WtaHashFunction:
		newLayer.wtaHasher = wta_hash.New(k*l, previousLayerNumOfNodes)

	case configuration.DensifiedWtaHashFunction:
		newLayer.binIds = make([]int, previousLayerNumOfNodes)
		newLayer.dwtaHasher =
			densified_wta_hash.New(k*l, previousLayerNumOfNodes)

	case configuration.DensifiedMinhashFunction:
		newLayer.minHasher = densified_minhash.New(k*l, previousLayerNumOfNodes)
		newLayer.binIds = newLayer.minHasher.GetMap(previousLayerNumOfNodes)

	case configuration.SparseRandomProjectionHashFunction:
		newLayer.srp =
			sparse_random_projection.New(previousLayerNumOfNodes, k*l, srpRatio)

	default:
		panic(fmt.Sprintf("Unexpected hash function %d.",
			configuration.Global.HashFunction))
	}

	if configuration.Global.LoadWeight {
		newLayer.weights = weights
		newLayer.bias = bias
		if configuration.Global.UseAdam {
			newLayer.adamAvgMom = adamAvgMom
			newLayer.adamAvgVel = adamAvgVel
		}
	} else {
		// TODO: check if normal dist is comparable to C++ implementation
		newLayer.weights = make([]float64, numOfNodes*previousLayerNumOfNodes)
		for i := range newLayer.weights {
			newLayer.weights[i] = rand.NormFloat64()*normalDistributionStdDev +
				normalDistributionMean
		}

		newLayer.bias = make([]float64, numOfNodes)
		for i := range newLayer.bias {
			newLayer.bias[i] = rand.NormFloat64()*normalDistributionStdDev +
				normalDistributionMean
		}

		if configuration.Global.UseAdam {
			size := numOfNodes * previousLayerNumOfNodes
			newLayer.adamAvgMom = make([]float64, size)
			newLayer.adamAvgVel = make([]float64, size)
		}
	}

	startTime := time.Now()

	newLayer.trainArray = make([]*node.NodeTrain, numOfNodes*batchSize)
	for i := range newLayer.trainArray {
		newLayer.trainArray[i] = node.NewNodeTrain(cowId)
	}

	// create nodes for this layer

	lastWeightsIndex := len(weights) - 1
	lastAdamAvgMomIndex := len(adamAvgMom) - 1
	lastAdamAvgVel := len(adamAvgVel) - 1

	// TODO: parallel!
	for i := range newLayer.nodes {
		newLayer.nodes[i] = newLayer.nodes[i].Update(
			cowId,
			previousLayerNumOfNodes,
			i,
			layerId,
			nodeType,
			batchSize,
			weights[previousLayerNumOfNodes*i:lastWeightsIndex],
			bias[i],
			adamAvgMom[previousLayerNumOfNodes*i:lastAdamAvgMomIndex],
			adamAvgVel[previousLayerNumOfNodes*i:lastAdamAvgVel],
			newLayer.trainArray,
		)
		newLayer.addToHashTable(
			cowId,
			newLayer.nodes[i].Weights(),
			// TODO: custom length? -> previousLayerNumOfNodes,
			newLayer.nodes[i].Bias(),
			i)
	}

	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	fmt.Printf("%d %v\n", numOfNodes, elapsedTime)

	if nodeType == node.Softmax {
		newLayer.normalizationConstants = make([]float64, batchSize)
	}

	return newLayer
}

// TODO: duplicated code from New
func (la *Layer) UpdateTable(cowId int) *Layer {
	l := la.cloneIfNeeded(cowId)

	switch configuration.Global.HashFunction {
	case configuration.WtaHashFunction:
		l.wtaHasher = wta_hash.New(l.k*l.l, l.previousLayerNumOfNodes)

	case configuration.DensifiedWtaHashFunction:
		l.binIds = make([]int, l.previousLayerNumOfNodes)
		l.dwtaHasher =
			densified_wta_hash.New(l.k*l.l, l.previousLayerNumOfNodes)

	case configuration.DensifiedMinhashFunction:
		l.minHasher = densified_minhash.New(l.k*l.l, l.previousLayerNumOfNodes)
		l.binIds = l.minHasher.GetMap(l.previousLayerNumOfNodes)

	case configuration.SparseRandomProjectionHashFunction:
		l.srp = sparse_random_projection.New(
			l.previousLayerNumOfNodes, l.k*l.l, srpRatio)

	default:
		panic(fmt.Sprintf("Unexpected hash function %d.",
			configuration.Global.HashFunction))
	}

	return l
}

func (l *Layer) UpdateRandomNodes() {
	swapRandNode := func(i, j int) {
		l.randNode[i], l.randNode[j] = l.randNode[j], l.randNode[i]
	}
	rand.Shuffle(l.numOfNodes, swapRandNode)
}

func (l *Layer) GetNodeById(nodeId int) *node.Node {
	return l.nodes[nodeId]
}

func (l *Layer) GetAllNodes() []*node.Node {
	return l.nodes
}

func (l *Layer) GetNodeCount() int {
	return l.numOfNodes
}

func (l *Layer) GetNomalizationConstant(inputId int) float64 {
	if l.nodeType != node.Softmax {
		panic("Call to GetNomalizationConstant for non-softmax layer")
	}
	return l.normalizationConstants[inputId]
}

func (l *Layer) addToHashTable(
	cowId int,
	weights []float64,
	bias float64,
	id int,
) {
	// LSH logic
	var hashes []int = nil

	switch configuration.Global.HashFunction {
	case configuration.WtaHashFunction:
		hashes = l.wtaHasher.GetHash(weights)

	case configuration.DensifiedWtaHashFunction:
		hashes = l.dwtaHasher.GetHashEasy(weights, topK)

	case configuration.DensifiedMinhashFunction:
		hashes = l.minHasher.GetHashEasy(l.binIds, weights, topK)

	case configuration.SparseRandomProjectionHashFunction:
		hashes = l.srp.GetHash(weights)

	default:
		panic(fmt.Sprintf("Unexpected hash function %d.",
			configuration.Global.HashFunction))
	}

	hashIndices := l.hashTables.HashesToIndex(hashes)
	bucketIndices := l.hashTables.Add(hashIndices, id+1)

	l.nodes[id] = l.nodes[id].SetIndices(cowId, hashIndices, bucketIndices)
}

func (l *Layer) cloneIfNeeded(cowId int) *Layer {
	if l.cowId != cowId {
		return l.clone(cowId)
	}
	return l
}

func (l *Layer) clone(cowId int) *Layer {
	// TODO: copy slices and pointers too?
	newLayer := &Layer{}
	*newLayer = *l
	newLayer.cowId = cowId
	return newLayer
}
