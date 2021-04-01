#!/bin/bash -e

wget ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2018/06/PXD010000/Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39.raw

# convert RAW files to MGF files
echo ThermoRawFileParser --input_directory=. --format=0 | make -f ../Makefile thermorawfileparser-noninteractive

rm ./*.raw
