// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layer

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/nlpodyssey/goslide/configuration"
	"github.com/nlpodyssey/goslide/densified_minhash"
	"github.com/nlpodyssey/goslide/densified_wta_hash"
	"github.com/nlpodyssey/goslide/index_value"
	"github.com/nlpodyssey/goslide/lsh"
	"github.com/nlpodyssey/goslide/node"
	"github.com/nlpodyssey/goslide/sparse_random_projection"
	"github.com/nlpodyssey/goslide/wta_hash"
)

const srpRatio = 32
const topK = 30
const normalDistributionStdDev = 0.01
const normalDistributionMean = 0.0
const threshold = 2
const mapLen = 325_056

type Layer struct {
	cowId                   int // "thread" ID for copy on write
	nodeType                node.NodeType
	nodes                   []*node.Node
	randNode                []int
	normalizationConstants  []float64
	k                       int
	l                       int
	previousLayerNumOfNodes int
	hashTables              *lsh.LSH
	wtaHasher               *wta_hash.WtaHash
	minHasher               *densified_minhash.DensifiedMinhash
	srp                     *sparse_random_projection.SparseRandomProjection
	dwtaHasher              *densified_wta_hash.DensifiedWtaHash
	binIds                  []int
}

type indexValuePairByValue []index_value.Pair

var _ sort.Interface = indexValuePairByValue{}

func (p indexValuePairByValue) Len() int      { return len(p) }
func (p indexValuePairByValue) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p indexValuePairByValue) Less(i, j int) bool {
	if p[i].Value == p[j].Value {
		return p[i].Index < p[j].Index
	}
	return p[i].Value < p[j].Value
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
		nodes:                   nil,
		randNode:                randNode,
		normalizationConstants:  nil,
		k:                       k,
		l:                       l,
		previousLayerNumOfNodes: previousLayerNumOfNodes,
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

	var (
		curWeights    []float64
		curBias       []float64
		curAdamAvgMom []float64
		curAdamAvgVel []float64
	)

	if configuration.Global.LoadWeight {
		curWeights = weights
		curBias = bias
		if configuration.Global.UseAdam {
			curAdamAvgMom = adamAvgMom
			curAdamAvgVel = adamAvgVel
		}
	} else {
		// TODO: check if normal dist is comparable to C++ implementation
		curWeights = make([]float64, numOfNodes*previousLayerNumOfNodes)
		for i := range curWeights {
			curWeights[i] = rand.NormFloat64()*normalDistributionStdDev +
				normalDistributionMean
		}

		curBias = make([]float64, numOfNodes)
		for i := range curBias {
			curBias[i] = rand.NormFloat64()*normalDistributionStdDev +
				normalDistributionMean
		}

		if configuration.Global.UseAdam {
			size := numOfNodes * previousLayerNumOfNodes
			curAdamAvgMom = make([]float64, size)
			curAdamAvgVel = make([]float64, size)
		}
	}

	startTime := time.Now()

	trainArray := make([]*node.NodeTrain, numOfNodes*batchSize)
	for i := range trainArray {
		trainArray[i] = node.NewNodeTrain(cowId)
	}

	// create nodes for this layer

	nodes := make([]*node.Node, numOfNodes)
	for i := range nodes {
		nodes[i] = node.NewEmptyNode(cowId)
	}

	// TODO: parallel!
	for i := range nodes {
		var layerAdamAvgMom []float64 = nil
		var layerAdamAvgVel []float64 = nil

		firstIndex := previousLayerNumOfNodes * i
		lastIndex := firstIndex + previousLayerNumOfNodes

		if configuration.Global.UseAdam {
			layerAdamAvgMom = curAdamAvgMom[firstIndex:lastIndex]
			layerAdamAvgVel = curAdamAvgVel[firstIndex:lastIndex]
		}

		nodes[i] = nodes[i].Update(
			cowId,
			previousLayerNumOfNodes,
			i,
			layerId,
			nodeType,
			batchSize,
			curWeights[firstIndex:lastIndex],
			curBias[i],
			layerAdamAvgMom,
			layerAdamAvgVel,
			trainArray[batchSize*i:batchSize*i+batchSize],
		)
		newLayer.addToHashTable(
			cowId,
			nodes[i].Weights(),
			// TODO: custom length? -> previousLayerNumOfNodes,
			nodes[i].Bias(),
			i)
	}

	newLayer.nodes = nodes

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
	rand.Shuffle(len(l.nodes), swapRandNode)
}

func (l *Layer) GetNodeById(nodeId int) *node.Node {
	return l.nodes[nodeId]
}

func (l *Layer) GetAllNodes() []*node.Node {
	return l.nodes
}

func (l *Layer) GetNomalizationConstant(inputId int) float64 {
	if l.nodeType != node.Softmax {
		panic("Call to GetNomalizationConstant for non-softmax layer")
	}
	return l.normalizationConstants[inputId]
}

func (la *Layer) QueryActiveNodeAndComputeActivations(
	cowId int,
	activeNodesPerLayer [][]index_value.Pair,
	layerIndex int,
	inputId int,
	label []int,
	sparsity float64,
) (int, *Layer) {
	l := la.cloneIfNeeded(cowId)

	currentLayerActiveNodes := activeNodesPerLayer[layerIndex]

	//LSH QueryLogic

	// Query out all the candidate nodes
	var length int
	in := 0

	if sparsity == 1.0 {
		length = len(l.nodes)
		// assuming not intitialized
		newActiveNodes := make([]index_value.Pair, length)
		for i := range newActiveNodes {
			newActiveNodes[i].Index = i
		}
		activeNodesPerLayer[layerIndex+1] = newActiveNodes
	} else {
		switch configuration.Global.LayerMode {
		case configuration.LayerMode1:
			var hashes []int = nil

			switch configuration.Global.HashFunction {
			case configuration.WtaHashFunction:
				hashes = l.wtaHasher.GetHash(currentLayerActiveNodes)

			case configuration.DensifiedWtaHashFunction:
				hashes = l.dwtaHasher.GetHash(currentLayerActiveNodes)

			case configuration.DensifiedMinhashFunction:
				hashes = l.minHasher.GetHashEasy(
					l.binIds, currentLayerActiveNodes, topK)

			case configuration.SparseRandomProjectionHashFunction:
				hashes = l.srp.GetHashSparse(currentLayerActiveNodes)

			default:
				panic(fmt.Sprintf("Unexpected hash function %d.",
					configuration.Global.HashFunction))
			}

			hashIndices := l.hashTables.HashesToIndex(hashes)
			actives := l.hashTables.RetrieveRaw(hashIndices)

			// Get candidates from hashtable

			counts := make(map[int]int)
			// Make sure that the true label node is in candidates
			if l.nodeType == node.Softmax && len(label) > 0 {
				for _, labelValue := range label {
					counts[labelValue] = l.l
				}
			}

			for _, iVal := range actives {
				for _, jVal := range iVal {
					counts[jVal-1] += 1
				}
			}

			// thresholding
			vect := make([]index_value.Pair, 0)
			for index, count := range counts {
				if count > threshold {
					vect = append(vect, index_value.Pair{Index: index})
				}
			}

			length = len(vect)
			activeNodesPerLayer[layerIndex+1] = make([]index_value.Pair, length)
			copy(activeNodesPerLayer[layerIndex+1], vect)
			in = length
		case configuration.LayerMode2:
			if l.nodeType == node.Softmax {
				length = int(math.Floor(float64(len(l.nodes)) * sparsity))
				activeNodesPerLayer[layerIndex+1] = make([]index_value.Pair, length)

				bs := make([]bool, mapLen) // bitset
				tmpSize := 0
				if l.nodeType == node.Softmax && len(label) > 0 {
					for i, labelValue := range label {
						activeNodesPerLayer[layerIndex+1][i].Index = labelValue
						bs[labelValue] = true
					}
					tmpSize = len(label)
				}
				for tmpSize < length {
					v := rand.Intn(len(l.nodes))
					if !bs[v] {
						activeNodesPerLayer[layerIndex+1][tmpSize].Index = v
						bs[v] = true
						tmpSize++
					}
				}
			}
		case configuration.LayerMode3:
			if l.nodeType == node.Softmax {
				length = int(math.Floor(float64(len(l.nodes)) * sparsity))
				activeNodesPerLayer[layerIndex+1] = make([]index_value.Pair, length)

				sortW := make([]index_value.Pair, 0)
				what := 0
				for s, curNode := range l.nodes {
					tmp := l.innerproduct(
						currentLayerActiveNodes, curNode.Weights())
					tmp += curNode.Bias()

					if intSliceContains(label, s) {
						sortW = append(sortW, index_value.Pair{
							Index: s,
							Value: -1000000000, // TODO: maybe min int?
						})
						what++
					} else {
						sortW = append(sortW, index_value.Pair{
							Index: s,
							Value: -tmp,
						})
					}
				}

				sort.Sort(indexValuePairByValue(sortW))

				for i, sw := range sortW {
					activeNodesPerLayer[layerIndex+1][i].Index = sw.Index
					if intSliceContains(label, sw.Index) {
						in = 1
					}
				}

			}
		case configuration.LayerMode4:
			// TODO: duplicate from above
			var hashes []int = nil

			switch configuration.Global.HashFunction {
			case configuration.WtaHashFunction:
				hashes = l.wtaHasher.GetHash(currentLayerActiveNodes)

			case configuration.DensifiedWtaHashFunction:
				hashes = l.dwtaHasher.GetHash(currentLayerActiveNodes)

			case configuration.DensifiedMinhashFunction:
				hashes = l.minHasher.GetHashEasy(
					l.binIds, currentLayerActiveNodes, topK)

			case configuration.SparseRandomProjectionHashFunction:
				hashes = l.srp.GetHashSparse(currentLayerActiveNodes)

			default:
				panic(fmt.Sprintf("Unexpected hash function %d.",
					configuration.Global.HashFunction))
			}

			hashIndices := l.hashTables.HashesToIndex(hashes)
			actives := l.hashTables.RetrieveRaw(hashIndices)

			// we now have a sparse array of indices of active nodes

			// Get candidates from hashtable

			counts := make(map[int]int)
			// Make sure that the true label node is in candidates
			if l.nodeType == node.Softmax && len(label) > 0 {
				for _, labelValue := range label {
					counts[labelValue] = l.l
				}
			}

			for _, iVal := range actives {
				// copy sparse array into (dense) map
				for _, jVal := range iVal {
					counts[jVal-1] += 1
				}
			}

			in = len(counts)

			if len(counts) < 1500 { // TODO: avoid magic number
				// TODO: it doesn't look like the best place to seed here
				rand.Seed(time.Now().UnixNano())
				start := rand.Intn(len(l.nodes))
				for i := start; i < len(l.nodes); i++ {
					if len(counts) >= 1000 { // TODO: avoid magic number
						break
					}
					cIndex := l.randNode[i]
					if _, ok := counts[cIndex]; !ok {
						counts[cIndex] = 0
					}
				}

				if len(counts) < 1000 { // TODO: avoid magic number
					for _, randNodeValue := range l.randNode {
						if len(counts) >= 1000 { // TODO: avoid magic number
							break
						}
						if _, ok := counts[randNodeValue]; !ok {
							counts[randNodeValue] = 0
						}
					}
				}
			}

			length = len(counts)
			newActiveNodes := make([]index_value.Pair, length)

			// copy map into new array
			i := 0
			for index := range counts {
				newActiveNodes[i].Index = index
				i++
			}
			activeNodesPerLayer[layerIndex+1] = newActiveNodes
		}
	}

	// ***********************************

	nextLayerActiveNodes := activeNodesPerLayer[layerIndex+1]

	maxValue := 0.0
	if l.nodeType == node.Softmax {
		l.normalizationConstants[inputId] = 0
	}

	nodes := l.nodes

	// find activation for all ACTIVE nodes in layer
	for i, pair := range nextLayerActiveNodes {
		var value float64
		value, nodes[pair.Index] =
			nodes[pair.Index].GetActivation(
				cowId, currentLayerActiveNodes, inputId,
			)
		nextLayerActiveNodes[i].Value = value
		if l.nodeType == node.Softmax && value > maxValue {
			maxValue = value
		}
	}

	if l.nodeType == node.Softmax {
		for i, pair := range nextLayerActiveNodes {
			realActivation := math.Exp(pair.Value - maxValue)
			nextLayerActiveNodes[i].Value = realActivation
			nodes[pair.Index] = nodes[pair.Index].SetlastActivation(
				cowId, inputId, realActivation,
			)
			l.normalizationConstants[inputId] += realActivation
		}
	}

	return in, l
}

func (l *Layer) ClearHashTables() {
	l.hashTables.Clear()
}

func (l *Layer) NumOfNodes() int {
	return len(l.nodes)
}

func (l *Layer) GetHashForInputProcessing(weights []float64) []int {
	// TODO: duplicated code
	switch configuration.Global.HashFunction {
	case configuration.WtaHashFunction:
		return l.wtaHasher.GetHashDense(weights)

	case configuration.DensifiedWtaHashFunction:
		return l.dwtaHasher.GetHashEasy(weights, topK)

	case configuration.DensifiedMinhashFunction:
		return l.minHasher.GetHashEasyDense(l.binIds, weights, topK)

	case configuration.SparseRandomProjectionHashFunction:
		return l.srp.GetHash(weights)

	default:
		panic(fmt.Sprintf("Unexpected hash function %d.",
			configuration.Global.HashFunction))
	}
}

func (l *Layer) HashesToIndex(hashes []int) []int {
	return l.hashTables.HashesToIndex(hashes)
}

func (l *Layer) HashTablesAdd(indices []int, id int) []int {
	return l.hashTables.Add(indices, id)
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
		hashes = l.wtaHasher.GetHashDense(weights)

	case configuration.DensifiedWtaHashFunction:
		hashes = l.dwtaHasher.GetHashEasy(weights, topK)

	case configuration.DensifiedMinhashFunction:
		hashes = l.minHasher.GetHashEasyDense(l.binIds, weights, topK)

	case configuration.SparseRandomProjectionHashFunction:
		hashes = l.srp.GetHash(weights)

	default:
		panic(fmt.Sprintf("Unexpected hash function %d.",
			configuration.Global.HashFunction))
	}

	hashIndices := l.hashTables.HashesToIndex(hashes)
	l.hashTables.Add(hashIndices, id+1)
}

func (l *Layer) innerproduct(
	activeNodesPerLayer []index_value.Pair,
	weights []float64,
) float64 {
	total := 0.0
	for _, pair := range activeNodesPerLayer {
		total += pair.Value * weights[pair.Index]
	}
	return total
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

func intSliceContains(slice []int, value int) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}
