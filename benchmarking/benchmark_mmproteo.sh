#!/bin/bash

echoerr() { echo "$@" 1>&2; }
TIME_CMD="/usr/bin/time --format=%K,%e"

echoerr "cores,operation,part,averageTotalMemoryInKB,elapsedRealTime"
for i in $(echo -e "1\n2\n4\n8"); do
	mkdir "$i"
	cd "$i" || (echoerr "could not change into $(pwd)/$i"; exit 1)
	pwd
	echo cores="$i"
	BASE_CMD="mmproteo -p PXD010000 --thread-count $i --log-to-stdout"
	echo download:
	for e in $(echo -e "raw\nmzid\nmzml"); do
		echo extension="$e"
		echoerr "$i,download,$e,$($TIME_CMD $BASE_CMD -n 8 -e "$e" download 3>&1 1>&2 2>&3)"
	done
	echo convertraw:
	echoerr "$i,convertraw,,$($TIME_CMD $BASE_CMD --thermo-output-format mgf convertraw 3>&1 1>&2 2>&3)"
	echo extract:
	echoerr "$i,extract,,$($TIME_CMD $BASE_CMD extract 3>&1 1>&2 2>&3)"
	echo mz2parquet:
	echoerr "$i,mz2parquet,,$($TIME_CMD $BASE_CMD mz2parquet 3>&1 1>&2 2>&3)"
	echo mgf2parquet:
	echoerr "$i,mgf2parquet,,$($TIME_CMD $BASE_CMD mgf2parquet 3>&1 1>&2 2>&3)"
	echo
	cd ..
done
