// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"os"
	"time"

	"github.com/nlpodyssey/goslide/configuration"
	"github.com/nlpodyssey/goslide/corpusio/xcrepo"
	"github.com/nlpodyssey/goslide/index_value"
	"github.com/nlpodyssey/goslide/network"
	"github.com/nlpodyssey/goslide/node"
)

var logger = log.New(os.Stderr, "", 0)

var globalTime time.Duration

func main() {
	const cowId = 0

	validateArguments()
	loadGlobalConfiguration()

	config := configuration.Global

	// Initialize Network

	numBatches := config.TotRecords / config.BatchSize
	numBatchesTest := config.TotRecordsTest / config.BatchSize

	layersTypes := make([]node.NodeType, config.NumLayer)
	for i := 0; i < config.NumLayer-1; i++ {
		layersTypes[i] = node.ReLU
	}
	layersTypes[config.NumLayer-1] = node.Softmax

	if config.LoadWeight {
		/*
			TODO: load weight...
			cnpy::npz_t arr;
			arr = cnpy::npz_load(Weights);
		*/
	}

	startTime := time.Now()
	myNet := network.New(
		cowId,
		config.SizesOfLayers,
		layersTypes,
		config.NumLayer,
		config.BatchSize,
		config.LearningRate,
		config.InputDim,
		config.K,
		config.L,
		config.RangePow,
		config.Sparsity,
	)
	endTime := time.Now()
	logger.Println("Network Initialization takes", endTime.Sub(startTime))

	// Start Training

	for e := 0; e < config.Epoch; e++ {
		logger.Println("Epoch", e)

		// train
		readDataSvm(cowId, numBatches, myNet, e)

		// test
		if e == config.Epoch-1 {
			evalDataSvm(cowId, numBatchesTest, myNet, (e+1)*numBatches)
		} else {
			evalDataSvm(cowId, 50, myNet, (e+1)*numBatches)
		}
	}
}

func readDataSvm(cowId, numBatches int, myNet *network.Network, epoch int) {
	config := configuration.Global

	file, err := os.Open(config.TrainData)
	if err != nil {
		logger.Fatal(err)
	}
	defer file.Close()

	scanner := xcrepo.NewScanner(file)
	if err := scanner.Err(); err != nil {
		logger.Fatal(err)
	}

	for i := 0; i < numBatches; i++ {
		if i > 0 && (i+epoch*numBatches)%config.Stepsize == 0 {
			evalDataSvm(cowId, 20, myNet, epoch*numBatches+i)
		}
		features := make([][]index_value.Pair, config.BatchSize)
		labels := make([][]int, config.BatchSize)

		for count := 0; count < config.BatchSize && scanner.Scan(); count++ {
			features[count] = scanner.Features()
			labels[count] = scanner.Labels()
		}

		if err := scanner.Err(); err != nil {
			logger.Fatalf("Error at line %d. %v", scanner.LineNumber(), err)
		}

		rehash := false
		rebuild := false

		if config.LayerMode == configuration.LayerMode1 || config.LayerMode == configuration.LayerMode4 {
			if (epoch*numBatches+i)%(config.Rehash/config.BatchSize) == (config.Rehash/config.BatchSize - 1) {
				rehash = true
			}
			// TODO: probably there was an error in the original code (using rehash at right)
			if (epoch*numBatches+i)%(config.Rebuild/config.BatchSize) == (config.Rebuild/config.BatchSize - 1) {
				rebuild = true
			}
		}

		startTime := time.Now()

		// logloss
		_, myNet = myNet.ProcessInput(
			cowId, features, labels, epoch*numBatches+i, rehash, rebuild)

		endTime := time.Now()
		globalTime += endTime.Sub(startTime)
	}
}

func evalDataSvm(cowId, numBatchesTest int, myNet *network.Network, iter int) {
	config := configuration.Global

	totCorrect := 0

	file, err := os.Open(config.TestData)
	if err != nil {
		logger.Fatal(err)
	}
	defer file.Close()

	scanner := xcrepo.NewScanner(file)
	if err := scanner.Err(); err != nil {
		logger.Fatal(err)
	}

	for i := 0; i < numBatchesTest; i++ {
		features := make([][]index_value.Pair, config.BatchSize)
		labels := make([][]int, config.BatchSize)
		numFeatures := 0
		numLabels := 0

		for count := 0; count < config.BatchSize && scanner.Scan(); count++ {
			features[count] = scanner.Features()
			labels[count] = scanner.Labels()

			numFeatures += len(features[count])
			numLabels += len(labels[count])
		}

		if err := scanner.Err(); err != nil {
			logger.Fatalf("Error at line %d. %v", scanner.LineNumber(), err)
		}

		logger.Println(config.BatchSize, "records, with", numFeatures,
			"features and", numLabels, "labels")

		var correctPredict int

		// FIXME: reassignment of myNet for CoW problematic
		correctPredict, myNet =
			myNet.PredictClass(cowId, features, labels)

		totCorrect += correctPredict

		logger.Println("Iter", i, "-",
			float64(totCorrect)/(float64(config.BatchSize)*(float64(i)+1)),
			"correct")
	}

	logger.Println("Over all:",
		float64(totCorrect)/(float64(numBatchesTest)*float64(config.BatchSize)),
		"correct")

	logger.Println(iter, globalTime,
		float64(totCorrect)/(float64(numBatchesTest)*float64(config.BatchSize)))
}

func validateArguments() {
	if len(os.Args) != 2 {
		logger.Println("Invalid or malformed arguments.")
		logger.Fatal("\nUsage:\n  goslide <json_configuration_file>\n")
	}
}

func loadGlobalConfiguration() {
	configFilename := os.Args[1]
	config, err := configuration.FromJsonFile(configFilename)

	if err != nil {
		logger.Println("An error occurred reading the configuration file.")
		logger.Fatal(err)
	}

	configuration.Global = config
}
