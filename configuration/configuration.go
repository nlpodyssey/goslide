// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package configuration

import (
	"encoding/json"
	"io/ioutil"
)

var Global = Default()

type Configuration struct {
	RangePow       []int
	K              []int
	L              []int
	Sparsity       []float64
	BatchSize      int
	Rehash         int
	Rebuild        int
	InputDim       int
	TotRecords     int
	TotRecordsTest int
	LearningRate   float64
	Epoch          int
	Stepsize       int
	SizesOfLayers  []int
	NumLayer       int
	TrainData      string
	TestData       string
	Weights        string
	SavedWeights   string
	LogFile        string
	UseAdam        bool
	HashFunction   HashFunctionType
	LoadWeight     bool
	LayerMode      LayerModeType
}

type HashFunctionType int8

const (
	WtaHashFunction HashFunctionType = iota + 1
	DensifiedWtaHashFunction
	DensifiedMinhashFunction
	SparseRandomProjectionHashFunction
)

type LayerModeType int8

const ( // TODO: find meaningful names
	LayerMode1 LayerModeType = iota + 1 // mode 1 == top-K treshold ?
	LayerMode2
	LayerMode3
	LayerMode4 // mode 4 == sampling ?
)

func Default() *Configuration {
	return &Configuration{
		RangePow:       make([]int, 0),
		K:              make([]int, 0),
		L:              make([]int, 0),
		Sparsity:       make([]float64, 0),
		BatchSize:      1000,
		Rehash:         1000,
		Rebuild:        1000,
		InputDim:       784,
		TotRecords:     60000,
		TotRecordsTest: 10000,
		LearningRate:   0.0001,
		Epoch:          5,
		Stepsize:       20,
		SizesOfLayers:  make([]int, 0),
		NumLayer:       3,
		TrainData:      "",
		TestData:       "",
		Weights:        "",
		SavedWeights:   "",
		LogFile:        "",
		UseAdam:        true,
		HashFunction:   DensifiedWtaHashFunction,
		LoadWeight:     false,
		LayerMode:      LayerMode4,
	}
}

func FromJsonFile(jsonFilename string) (*Configuration, error) {
	bytes, err := ioutil.ReadFile(jsonFilename)
	if err != nil {
		return nil, err
	}

	config := Default()
	err = json.Unmarshal(bytes, config)
	if err != nil {
		return nil, err
	}

	return config, nil
}
