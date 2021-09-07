#!/bin/bash

BASE_PATH=$(realpath $(dirname $0))

mkdir downloaded
cd downloaded
$BASE_PATH/download.sh
cd ..

for run in `seq 1 5`
do mkdir "run_$run"
cd "run_$run"
$BASE_PATH/benchmark_mmproteo.sh 3>&1 1>&2 2>&3 | tee -a benchmark.csv
cd ..
done
