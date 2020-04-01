// Copyright (c) 2020, The GoSLIDE Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"os"

	"github.com/nlpodyssey/goslide/configuration"
)

var logger = log.New(os.Stderr, "", 0)

func main() {
	validateArguments()
	loadGlobalConfiguration()

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
