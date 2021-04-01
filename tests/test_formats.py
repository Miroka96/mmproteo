import os

from mmproteo.utils import filters
from mmproteo.utils.formats import mz, read
from .utils import defaults
from .utils.fixtures import run_with_datasets


def test_read_mzid(run_with_datasets):
    read.read(defaults.MZID_FILE_PATH)


def test_merge_mzml_mzid_to_parquet(run_with_datasets):
    if os.path.isfile(defaults.MZMLID_FILE_PATH):
        os.remove(defaults.MZMLID_FILE_PATH)
    parquet_files = mz.merge_mzml_and_mzid_files_to_parquet(filenames=[defaults.MZML_FILE_PATH,
                                                                       defaults.MZID_FILE_PATH])
    assert os.path.isfile(defaults.MZMLID_FILE_PATH)
    assert parquet_files == [defaults.MZMLID_FILE_PATH]
    os.remove(defaults.MZMLID_FILE_PATH)


def test_filter_files_list():
    filenames = [
        "abc.txt",
        "def.txt.gz",
        "ghi.gz"
    ]

    assert ["abc.txt"] == filters.filter_files_list(filenames=filenames, max_num_files=1)
    assert ["abc.txt", "def.txt.gz"] == filters.filter_files_list(filenames=filenames, file_extensions=["txt"])
    assert ["def.txt.gz", "ghi.gz"] == filters.filter_files_list(filenames=filenames, file_extensions=["gz"])
    assert ["abc.txt", "def.txt.gz", "ghi.gz"] == filters.filter_files_list(filenames=filenames,
                                                                            file_extensions=["gz", "txt"])
