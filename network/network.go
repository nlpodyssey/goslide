// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package network

import (
	"fmt"
	"math"
	"time"

	"github.com/nlpodyssey/goslide/configuration"
	"github.com/nlpodyssey/goslide/index_value"
	"github.com/nlpodyssey/goslide/layer"
	"github.com/nlpodyssey/goslide/node"
)

const (
	beta1 = 0.9
	beta2 = 0.999
	eps   = 0.00000001
	debug = true
)

type Network struct {
	cowId            int // "thread" ID for copy on write
	hiddenLayers     []*layer.Layer
	learningRate     float64
	numberOfLayers   int
	sizesOfLayers    []int
	layersTypes      []node.NodeType
	sparsity         []float64
	currentBatchSize int
}

func New(
	cowId int,
	sizesOfLayers []int,
	layerTypes []node.NodeType,
	numOfLayers int,
	batchSize int,
	lr float64,
	inputDim int,
	k []int,
	l []int,
	rangePow []int,
	sparsity []float64,
) *Network {
	hiddenLayers := make([]*layer.Layer, numOfLayers)
	{
		var (
			weight     []float64
			bias       []float64
			adamAvgMom []float64
			adamAvgVel []float64
		)

		if configuration.Global.LoadWeight {
			/*
				TODO: load weight...
				weightArr = arr["w_layer_"+to_string(i)];
				weight = weightArr.data<float>();
				biasArr = arr["b_layer_"+to_string(i)];
				bias = biasArr.data<float>();

				adamArr = arr["am_layer_"+to_string(i)];
				adamAvgMom = adamArr.data<float>();
				adamvArr = arr["av_layer_"+to_string(i)];
				adamAvgVel = adamvArr.data<float>();
			*/
		}

		hiddenLayers[0] = layer.New(
			cowId,
			sizesOfLayers[0],
			inputDim,
			0,
			layerTypes[0],
			batchSize,
			k[0],
			l[0],
			rangePow[0],
			sparsity[0],
			weight,
			bias,
			adamAvgMom,
			adamAvgVel,
		)
	}

	for i := 1; i < numOfLayers; i++ {
		var (
			weight     []float64
			bias       []float64
			adamAvgMom []float64
			adamAvgVel []float64
		)

		if configuration.Global.LoadWeight {
			/*
				TODO: load weight...

				weightArr = arr["w_layer_"+to_string(i)];
				weight = weightArr.data<float>();
				biasArr = arr["b_layer_"+to_string(i)];
				bias = biasArr.data<float>();

				adamArr = arr["am_layer_"+to_string(i)];
				adamAvgMom = adamArr.data<float>();
				adamvArr = arr["av_layer_"+to_string(i)];
				adamAvgVel = adamvArr.data<float>();
			*/
		}

		hiddenLayers[i] = layer.New(
			cowId,
			sizesOfLayers[i],
			sizesOfLayers[i-1],
			i,
			layerTypes[i],
			batchSize,
			k[i],
			l[i],
			rangePow[i],
			sparsity[i],
			weight,
			bias,
			adamAvgMom,
			adamAvgVel,
		)
	}

	return &Network{
		cowId:            cowId,
		hiddenLayers:     hiddenLayers,
		learningRate:     lr,
		numberOfLayers:   numOfLayers,
		sizesOfLayers:    sizesOfLayers,
		layersTypes:      layerTypes,
		sparsity:         sparsity,
		currentBatchSize: batchSize,
	}
}

func (n *Network) GetLayer(layerId int) *layer.Layer {
	return n.hiddenLayers[layerId]
}

func (ne *Network) PredictClass(
	cowId int,
	features [][]index_value.Pair,
	labels [][]int,
) (int, *Network) {
	n := ne.cloneIfNeeded(cowId)

	startTime := time.Now()
	correctPred := 0

	// TODO: parallel!
	for i := 0; i < n.currentBatchSize; i++ {
		activeNodesPerLayer := make([][]index_value.Pair, n.numberOfLayers+1)
		activeNodesPerLayer[0] = features[i]

		//inference
		for j := 0; j < n.numberOfLayers; j++ {
			_, n.hiddenLayers[j] =
				n.hiddenLayers[j].QueryActiveNodeAndComputeActivations(
					cowId,
					activeNodesPerLayer,
					j,
					i,
					[]int{},
					n.sparsity[n.numberOfLayers+j],
					-1,
				)
		}

		//compute softmax
		maxAct := -222222222.0 // TODO: ...
		predictClass := -1
		for _, pair := range activeNodesPerLayer[n.numberOfLayers] {
			curAct := n.hiddenLayers[n.numberOfLayers-1].GetNodeById(
				pair.Index).GetLastActivation(i)
			if maxAct < curAct {
				maxAct = curAct
				predictClass = pair.Index
			}
		}

		if intSliceContains(labels[i], predictClass) {
			correctPred++
		}
	}

	endTime := time.Now()
	fmt.Printf("Inference takes %v.\n", endTime.Sub(startTime))

	return correctPred, n
}

func (ne *Network) ProcessInput(
	cowId int,
	features [][]index_value.Pair,
	labels [][]int,
	iter int,
	rehash bool,
	rebuild bool,
) (float64, *Network) {
	n := ne.cloneIfNeeded(cowId)

	logLoss := 0.0
	avgRetrieval := make([]int, n.numberOfLayers)
	// avgRetrieval contains all zeroes by default

	if iter%6946 == 6945 { // TODO: avoid magic number
		// TODO: ?? _learningRate *= 0.5;

		for i := 1; i < n.numberOfLayers; i++ {
			// FIXME: or should this be done only on the very last layer?
			n.hiddenLayers[i].UpdateRandomNodes()
		}
	}

	tmpLr := n.learningRate
	if configuration.Global.UseAdam {
		tmpLr = n.learningRate *
			math.Sqrt(1-math.Pow(beta2, float64(iter)+1.0)) /
			(1.0 - math.Pow(beta1, float64(iter)+1.0))
		// TODO: ?? else: tmplr *= pow(0.9, iter/10.0);
	}

	activeNodesPerBatch := make([][][]index_value.Pair, n.currentBatchSize)

	// TODO: parallel!
	for i := 0; i < n.currentBatchSize; i++ {
		activeNodesPerLayer := make([][]index_value.Pair, n.numberOfLayers+1)
		activeNodesPerBatch[i] = activeNodesPerLayer
		activeNodesPerLayer[0] = features[i]

		for j := 0; j < n.numberOfLayers; j++ {
			var in int
			in, n.hiddenLayers[j] = n.hiddenLayers[j].QueryActiveNodeAndComputeActivations(
				cowId,
				activeNodesPerLayer,
				j,
				i,
				labels[i],
				n.sparsity[j],
				iter*n.currentBatchSize+i,
			)
			avgRetrieval[j] += in
		}

		// Now backpropagate.
		// layers
		for j := n.numberOfLayers - 1; j >= 0; j-- {
			layer := n.hiddenLayers[j]
			// nodes
			for _, pair := range activeNodesPerBatch[i][j+1] {
				node := layer.GetNodeById(pair.Index)
				if j == n.numberOfLayers-1 {
					//TODO: Compute Extra stats: labels[i];
					// FIXME: we should reassign the cow-ed node into the layer
					node.ComputeExtaStatsForSoftMax(
						cowId,
						layer.GetNomalizationConstant(i),
						i,
						labels[i],
					)
				}
				if j != 0 {
					prevLayer := n.hiddenLayers[j-1]
					// FIXME: we should reassign the cow-ed node into the layer
					allNodes := prevLayer.GetAllNodes()
					node.BackPropagate(
						cowId,
						&allNodes, // FIXME: problematic...
						activeNodesPerBatch[i][j],
						tmpLr,
						i,
					)
				} else {
					// FIXME: we should reassign the cow-ed node into the layer
					node.BackPropagateFirstLayer(
						cowId,
						features[i],
						tmpLr,
						i,
					)
				}
			}
		}
	}

	for l := 0; l < n.numberOfLayers; l++ {
		tmpRehash := rehash && n.sparsity[l] < 1.0
		tmpRebuild := rebuild && n.sparsity[l] < 1.0

		if tmpRehash {
			n.hiddenLayers[l].ClearHashTables()
		}

		if tmpRebuild {
			n.hiddenLayers[l] = n.hiddenLayers[l].UpdateTable(cowId)
		}

		const ratio = 1

		// TODO: parallel!
		for m := 0; m < n.hiddenLayers[l].NumOfNodes(); m++ {
			tmp := n.hiddenLayers[l].GetNodeById(m)
			dim := tmp.Dim()
			curWeights := tmp.Weights()

			if configuration.Global.UseAdam {
				for d := 0; d < dim; d++ {
					t := tmp.GetT(d)
					mom := tmp.GetAdamAvgMom(d)
					vel := tmp.GetAdamAvgVel(d)

					mom = beta1*mom + (1-beta1)*t
					vel = beta2*vel + (1-beta2)*t*t

					// Direclty modify the weight (by reference)
					curWeights[d] += ratio * tmpLr * mom / (math.Sqrt(vel) + eps)
					tmp = tmp.SetAdamAvgMom(cowId, d, mom)
					tmp = tmp.SetAdamAvgVel(cowId, d, vel)
					tmp = tmp.SetT(cowId, d, 0)
				}

				tmp = tmp.SetAdamAvgMomBias(cowId,
					beta1*tmp.GetAdamAvgMomBias()+(1-beta1)*tmp.GetTBias())
				tmp = tmp.SetAdamAvgVelBias(cowId,
					beta2*tmp.GetAdamAvgVelBias()+(1-beta2)*tmp.GetTBias()*tmp.GetTBias())
				tmp = tmp.SetBias(cowId,
					tmp.GetBias()+ratio*tmpLr*tmp.GetAdamAvgMomBias()/(math.Sqrt(tmp.GetAdamAvgVelBias())+eps))
				tmp = tmp.SetTBias(cowId, 0)
			} else {
				tmp = tmp.CopyWeightsAndBiasFromMirror(cowId)
			}

			// FIXME: now tmp was modified and should be set back into hiddenLayers[l]
			if tmpRehash {
				hashes := n.hiddenLayers[l].GetHashForInputProcessing(curWeights)
				hashIndices := n.hiddenLayers[l].HashesToIndex(hashes)
				n.hiddenLayers[l].HashTablesAdd(hashIndices, m+1)
			}
		}

	}

	if debug && rehash {
		for i, v := range avgRetrieval {
			fmt.Printf("Avg sample size [%d] = %f\n",
				i, float64(v)/float64(n.currentBatchSize))
		}
	}

	return logLoss, n
}

func (n *Network) cloneIfNeeded(cowId int) *Network {
	if n.cowId != cowId {
		return n.clone(cowId)
	}
	return n
}

func (n *Network) clone(cowId int) *Network {
	// TODO: copy slices too?
	newNetwork := &Network{}
	*newNetwork = *n
	newNetwork.cowId = cowId
	return newNetwork
}

func intSliceContains(slice []int, value int) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}
