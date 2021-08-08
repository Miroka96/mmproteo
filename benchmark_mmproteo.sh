#!/bin/bash

echoerr() { echo "$@" 1>&2; }

for i in $(echo -e "1\n4\n16"); do
	mkdir $i
	cd $i
	echoerr cores=$i
	BASE_CMD="mmproteo -p PXD010000 --thread-count $i --log-to-stdout"
	echoerr download:
	for e in $(echo -e "mgf\nmzid\nmzml"); do
		echo extension=$e
		time $BASE_CMD -n 64 -e $e download
	done
	echoerr extract:
	time $BASE_CMD extract
	echoerr mz2parquet:
	time $BASE_CMD mz2parquet
	echoerr mgf2parquet:
	time $BASE_CMD mgf2parquet
	echoerr
	cd ..
	sleep 3600
done