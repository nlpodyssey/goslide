// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package network

import (
	"fmt"
	"math"
	"time"

	"github.com/nlpodyssey/goslide/configuration"
	"github.com/nlpodyssey/goslide/dataset"
	"github.com/nlpodyssey/goslide/index_value"
	"github.com/nlpodyssey/goslide/layer"
	"github.com/nlpodyssey/goslide/node"
)

const (
	beta1 = 0.9
	beta2 = 0.999
	eps   = 0.00000001
)

type Network struct {
	cowId          int // "thread" ID for copy on write
	hiddenLayers   []*layer.Layer
	learningRate   float64
	numberOfLayers int
	sparsity       []float64
}

func New(
	cowId int,
	numOfLayers int,
	sizesOfLayers []int,
	layerTypes []node.NodeType,
	batchSize int,
	learningRate float64,
	inputDim int,
	k []int,
	l []int,
	rangePow []int,
	sparsity []float64,
) *Network {
	hiddenLayers := make([]*layer.Layer, numOfLayers)
	previousLayerNumOfNodes := inputDim

	for i := range hiddenLayers {
		var (
			weight     []float64 = nil
			bias       []float64 = nil
			adamAvgMom []float64 = nil
			adamAvgVel []float64 = nil
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
			previousLayerNumOfNodes,
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

		previousLayerNumOfNodes = sizesOfLayers[i]
	}

	return &Network{
		cowId:          cowId,
		hiddenLayers:   hiddenLayers,
		learningRate:   learningRate,
		numberOfLayers: numOfLayers,
		sparsity:       sparsity,
	}
}

func (ne *Network) PredictClass(
	cowId int,
	examples []dataset.Example,
) (int, *Network) {
	n := ne.cloneIfNeeded(cowId)

	startTime := time.Now()
	correctPred := 0

	// TODO: parallel!
	for i, example := range examples {
		activeNodesPerLayer := make([][]index_value.Pair, n.numberOfLayers+1)
		activeNodesPerLayer[0] = example.Features

		//inference
		for layerIndex, layer := range n.hiddenLayers {
			_, n.hiddenLayers[layerIndex] =
				layer.QueryActiveNodeAndComputeActivations(
					cowId,
					activeNodesPerLayer,
					layerIndex,
					i,
					[]int{},
					n.sparsity[n.numberOfLayers+layerIndex],
				)
		}

		//compute softmax
		lastHiddenLayer := n.hiddenLayers[n.numberOfLayers-1]
		var maxAct float64
		var predictClass int
		for pairIndex, pair := range activeNodesPerLayer[n.numberOfLayers] {
			act := lastHiddenLayer.GetNodeById(pair.Index).GetLastActivation(i)
			if maxAct < act || pairIndex == 0 {
				maxAct = act
				predictClass = pair.Index
			}
		}

		if intSliceContains(example.Labels, predictClass) {
			correctPred++
		}
	}

	endTime := time.Now()
	fmt.Printf("Inference takes %v.\n", endTime.Sub(startTime))

	return correctPred, n
}

func (ne *Network) ProcessInput(
	cowId int,
	examples []dataset.Example,
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

	// TODO: parallel!
	for i, example := range examples {
		activeNodesPerLayer := make([][]index_value.Pair, n.numberOfLayers+1)
		activeNodesPerLayer[0] = example.Features

		for layerIndex, layer := range n.hiddenLayers {
			var in int
			in, n.hiddenLayers[layerIndex] =
				layer.QueryActiveNodeAndComputeActivations(
					cowId,
					activeNodesPerLayer,
					layerIndex,
					i,
					example.Labels,
					n.sparsity[layerIndex],
				)
			avgRetrieval[layerIndex] += in
		}

		// Backpropagation
		for layerIndex := n.numberOfLayers - 1; layerIndex >= 0; layerIndex-- {
			layer := n.hiddenLayers[layerIndex]
			// nodes
			for _, pair := range activeNodesPerLayer[layerIndex+1] {
				node := layer.GetNodeById(pair.Index)
				if layerIndex == n.numberOfLayers-1 {
					//TODO: Compute Extra stats: labels[i];
					// FIXME: we should reassign the cow-ed node into the layer
					node.ComputeExtaStatsForSoftMax(
						cowId,
						layer.GetNomalizationConstant(i),
						i,
						example.Labels,
					)
				}
				if layerIndex != 0 {
					prevLayer := n.hiddenLayers[layerIndex-1]
					// FIXME: we should reassign the cow-ed node into the layer
					allNodes := prevLayer.GetAllNodes()
					node.BackPropagate(
						cowId,
						&allNodes, // FIXME: problematic...
						activeNodesPerLayer[layerIndex],
						tmpLr,
						i,
					)
				} else {
					// FIXME: we should reassign the cow-ed node into the layer
					node.BackPropagateFirstLayer(
						cowId,
						example.Features,
						tmpLr,
						i,
					)
				}
			}
		}
	}

	for layerIndex, layer := range n.hiddenLayers {
		tmpRehash := rehash && n.sparsity[layerIndex] < 1.0
		tmpRebuild := rebuild && n.sparsity[layerIndex] < 1.0

		if tmpRehash {
			layer.ClearHashTables()
		}

		if tmpRebuild {
			n.hiddenLayers[layerIndex] = layer.UpdateTable(cowId)
		}

		const ratio = 1

		// TODO: parallel!
		for m := 0; m < layer.NumOfNodes(); m++ {
			tmp := layer.GetNodeById(m)
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
				hashes := layer.GetHashForInputProcessing(curWeights)
				hashIndices := layer.HashesToIndex(hashes)
				layer.HashTablesAdd(hashIndices, m+1)
			}
		}

	}

	if rehash {
		fmt.Printf("Avg sample size")
		for _, v := range avgRetrieval {
			fmt.Printf(" %.3f", float64(v)/float64(len(examples)))
		}
		fmt.Println()
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
