import os

from .utils.fixtures import run_in_temp_directory
from .utils.defaults import DEFAULT_PROJECT

MZ_PIPELINE_COMMAND = f"mmproteo -p {DEFAULT_PROJECT} -e mzid,mzml -n 2 -c fileName,fileSizeBytes " \
                      f"list download extract mz2parquet"


def test_mmproteo_download(run_in_temp_directory):
    os.system(MZ_PIPELINE_COMMAND)
    # TODO
