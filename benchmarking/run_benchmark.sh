#!/bin/bash

BASE_PATH=$(realpath $(dirname $0))

mkdir downloaded
cd downloaded
$BASE_PATH/download.sh
cd ..
$BASE_PATH/benchmark_mmproteo.sh 3>&1 1>&2 2>&3 | tee -a benchmark.csv
