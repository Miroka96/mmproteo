#!/bin/bash

echo "####################################################"
echo "#THIS SCRIPT MUST RUN ONCE WITHOUT FAILED DOWNLOADS#"
echo "####################################################"
echo
mmproteo -p PXD010000 -n 8 -e mzml --count-failed-files download
mmproteo -p PXD010000 -n 8 -e mzid --count-failed-files download
mmproteo -p PXD010000 -n 8 -e raw --count-failed-files download
