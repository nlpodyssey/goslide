// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// I/O support for The Extreme Classification Repository
// Multi-label Datasets sparse format.
//
// http://manikvarma.org/downloads/XC/XMLRepository.html
package xcrepo

import (
	"bufio"
	"errors"
	"io"
	"strconv"
	"strings"
)

type Scanner struct {
	bufScanner     *bufio.Scanner
	err            error
	lineNumber     int
	totalPoints    int
	numFeatures    int
	numLabels      int
	labels         []int
	featureIndices []int
	featureValues  []float64
}

// Errors returned by Scanner.
var (
	ErrMalformedHeader         = errors.New("xcrepo.Scanner: malformed or missing header")
	ErrNotEnoughFields         = errors.New("xcrepo.Scanner: not enough fields")
	ErrMalformedLabels         = errors.New("xcrepo.Scanner: malformed labels")
	ErrMissingFeatures         = errors.New("xcrepo.Scanner: missing features")
	ErrMalformedFeatures       = errors.New("xcrepo.Scanner: malformed features")
	ErrLabelOutOfBounds        = errors.New("xcrepo.Scanner: label value out of bounds")
	ErrFeatureIndexOutOfBounds = errors.New("xcrepo.Scanner: feature index out of bounds")
)

func NewScanner(r io.Reader) *Scanner {
	s := &Scanner{
		bufScanner: bufio.NewScanner(r),
	}

	if s.bufScanner.Err() != nil {
		return s
	}

	s.scanHeader()

	return s
}

// Err returns the first non-EOF error that was encountered by the Scanner.
func (s *Scanner) Err() error {
	if s.err != nil {
		return s.err
	}
	return s.bufScanner.Err()
}

func (s *Scanner) LineNumber() int {
	return s.lineNumber
}

func (s *Scanner) TotalPoints() int {
	return s.totalPoints
}

func (s *Scanner) NumFeatures() int {
	return s.numFeatures
}

func (s *Scanner) NumLabels() int {
	return s.numLabels
}

func (s *Scanner) Labels() []int {
	return s.labels
}

func (s *Scanner) FeatureIndices() []int {
	return s.featureIndices
}

func (s *Scanner) FeatureValues() []float64 {
	return s.featureValues
}

func (s *Scanner) FeaturesLength() int {
	return len(s.featureIndices)
}

func (s *Scanner) Scan() bool {
	if s.Err() != nil {
		return false
	}

	if ok := s.bufScanner.Scan(); !ok || s.bufScanner.Err() != nil {
		return false
	}
	s.lineNumber++
	text := s.bufScanner.Text()

	values := strings.Split(text, " ")
	if len(values) == 0 {
		s.err = ErrNotEnoughFields
		return false
	}

	if ok := s.parseLabels(values[0]); !ok {
		return false
	}
	if ok := s.parseFeatures(values[1:len(values)]); !ok {
		return false
	}

	return true
}

func (s *Scanner) parseLabels(str string) bool {
	if len(str) == 0 {
		s.labels = make([]int, 0)
		return true
	}

	labels := strings.Split(str, ",")
	s.labels = make([]int, len(labels))

	for i, value := range labels {
		label, err := strconv.Atoi(value)
		if err != nil {
			s.err = ErrMalformedLabels
			return false
		}
		if label < 0 || label >= s.numLabels {
			s.err = ErrLabelOutOfBounds
			return false
		}
		s.labels[i] = label
	}
	return true
}

func (s *Scanner) parseFeatures(pairs []string) bool {
	lenPairs := len(pairs)
	if lenPairs == 0 {
		s.err = ErrMissingFeatures
		return false
	}

	s.featureIndices = make([]int, lenPairs)
	s.featureValues = make([]float64, lenPairs)

	for i, pair := range pairs {
		splitPair := strings.Split(pair, ":")
		if len(splitPair) != 2 {
			s.err = ErrMalformedFeatures
			return false
		}

		featureIndex, err := strconv.Atoi(splitPair[0])
		if err != nil {
			s.err = ErrMalformedFeatures
			return false
		}
		if featureIndex < 0 || featureIndex >= s.NumFeatures() {
			s.err = ErrFeatureIndexOutOfBounds
			return false
		}

		featureValue, err := strconv.ParseFloat(splitPair[1], 64)
		if err != nil {
			s.err = ErrMalformedFeatures
			return false
		}

		s.featureIndices[i] = featureIndex
		s.featureValues[i] = featureValue
	}

	return true
}

func (s *Scanner) scanHeader() {
	s.bufScanner.Scan()
	if s.bufScanner.Err() != nil {
		return
	}
	s.lineNumber++
	text := s.bufScanner.Text()

	values := strings.Split(text, " ")
	if len(values) != 3 {
		s.err = ErrMalformedHeader
		return
	}

	var ints [3]int
	for i, value := range values {
		var err error
		ints[i], err = strconv.Atoi(value)
		if err != nil || ints[i] <= 0 {
			s.err = ErrMalformedHeader
			return
		}
	}

	s.totalPoints = ints[0]
	s.numFeatures = ints[1]
	s.numLabels = ints[2]
}
