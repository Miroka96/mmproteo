#!/bin/bash

$(dirname $0)/benchmark_mmproteo.sh 3>&1 1>&2 2>&3 | tee -a benchmark.log
