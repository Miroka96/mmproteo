#!/bin/bash

echoerr() { echo -ne "$@" 1>&2; }
TIME_CMD="/usr/bin/time --format=%K,%e"

echoerr "cores,operation,part,averageTotalMemoryInKB,elapsedRealTime\n"
for i in $(echo -e "1\n2\n4\n8" | shuf); do
	mkdir "$i"
	cd "$i" || (echoerr "could not change into $(pwd)/$i"; exit 1)
	pwd
	echo cores="$i"
	BASE_CMD="mmproteo -p PXD010000 --thread-count $i --log-to-stdout"
	echo download:
	for e in $(echo -e "raw\nmzid\nmzml"); do
		echo extension="$e"
		echoerr "$i,download,$e,"
		$TIME_CMD $BASE_CMD -n 8 -e "$e" download
	done
	rm ./*.{gz,raw}
	cp ../downloaded/*.{gz,raw} .
	echo convertraw:
	echoerr "$i,convertraw,,"
	$TIME_CMD $BASE_CMD --thermo-output-format mgf convertraw
	echo extract:
	echoerr "$i,extract,,"
	$TIME_CMD $BASE_CMD extract
	echo mz2parquet:
	echoerr "$i,mz2parquet,,"
	$TIME_CMD $BASE_CMD mz2parquet
	echo mgf2parquet:
	echoerr "$i,mgf2parquet,,"
	$TIME_CMD $BASE_CMD mgf2parquet
	echo
	cd ..
done
