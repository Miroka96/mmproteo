#!/usr/bin/python3
import mmproteo.utils.filters
from mmproteo.utils import pride, formats
import pandas as pd
try:
    from utils.defaults import *
except ModuleNotFoundError:
    from .utils.defaults import *


def store_get_project_info():
    with open(f"resources/{DEFAULT_PROJECT}_info.txt", "w") as file:
        file.write(pride.get_project_info(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API]))


def test_get_project_info():
    with open(f"resources/{DEFAULT_PROJECT}_info.txt", "r") as file:
        expected_info = file.read()

    received_info = pride.get_project_info(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    assert expected_info == received_info


def store_get_project_files():
    pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])\
        .to_parquet(f"resources/{DEFAULT_PROJECT}_files.parquet")


def test_get_project_files():
    expected_files = pd.read_parquet(f"resources/{DEFAULT_PROJECT}_files.parquet")
    received_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    pd._testing.assert_frame_equal(expected_files, received_files, check_exact=True)


def store_get_raw_project_files():
    project_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    project_files = mmproteo.utils.filters.filter_files_df(files_df=project_files, file_extensions={"raw"})
    project_files.to_parquet(f"resources/{DEFAULT_PROJECT}_files_raw.parquet")


def test_list_raw_project_files():
    expected_files = pd.read_parquet(f"resources/{DEFAULT_PROJECT}_files_raw.parquet")
    received_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    received_files = mmproteo.utils.filters.filter_files_df(files_df=received_files, file_extensions={"raw"})
    pd._testing.assert_frame_equal(expected_files, received_files, check_exact=True)


def store_get_mgf_mzid_project_files():
    project_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    project_files = mmproteo.utils.filters.filter_files_df(files_df=project_files, file_extensions={"mgf", "mzid"})
    project_files.to_parquet(f"resources/{DEFAULT_PROJECT}_files_mgf_mzid.parquet")


def test_list_mgf_mzid_files():
    expected_files = pd.read_parquet(f"resources/{DEFAULT_PROJECT}_files_mgf_mzid.parquet")
    received_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    received_files = mmproteo.utils.filters.filter_files_df(files_df=received_files, file_extensions={"mgf", "mzid"})
    pd._testing.assert_frame_equal(expected_files, received_files, check_exact=True)


def store_get_gz_project_files():
    project_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    project_files = mmproteo.utils.filters.filter_files_df(files_df=project_files, file_extensions={"gz"})
    project_files.to_parquet(f"resources/{DEFAULT_PROJECT}_files_gz.parquet")


def test_list_gz_files():
    expected_files = pd.read_parquet(f"resources/{DEFAULT_PROJECT}_files_gz.parquet")
    received_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    received_files = mmproteo.utils.filters.filter_files_df(files_df=received_files, file_extensions={"gz"})
    pd._testing.assert_frame_equal(expected_files, received_files, check_exact=True)


if __name__ == '__main__':
    store_get_project_info()
    store_get_project_files()
    store_get_raw_project_files()
    store_get_mgf_mzid_project_files()
    store_get_gz_project_files()
