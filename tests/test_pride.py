#!/usr/bin/python3
import os
from typing import NoReturn, Optional

import pytest

import mmproteo.utils.filters
from mmproteo.utils import pride
import pandas as pd

try:
    from utils.defaults import DEFAULT_TEST_API, DEFAULT_PROJECT, RESOURCES_PATH
except ModuleNotFoundError:
    from .utils.defaults import DEFAULT_TEST_API, DEFAULT_PROJECT


def store_get_project_info() -> None:
    with open(os.path.join(RESOURCES_PATH, f"{DEFAULT_PROJECT}_info.txt"), "w") as file:
        project_info = pride.get_project_info(project_name=DEFAULT_PROJECT,
                                              api_versions=[DEFAULT_TEST_API])
        assert project_info is not None, "project info should be available"
        file.write(project_info)


def test_get_project_info(run_in_resources_directory: pytest.Function) -> Optional[NoReturn]:
    with open(f"{DEFAULT_PROJECT}_info.txt", "r") as file:
        expected_info = file.read()

    received_info = pride.get_project_info(project_name=DEFAULT_PROJECT,
                                           api_versions=[DEFAULT_TEST_API])
    assert expected_info == received_info
    return None


def store_get_project_files() -> Optional[NoReturn]:
    project_files = pride.get_project_files(project_name=DEFAULT_PROJECT,
                                            api_versions=[DEFAULT_TEST_API])
    assert project_files is not None, "project files should be available"
    project_files.to_parquet(os.path.join(RESOURCES_PATH, f"{DEFAULT_PROJECT}_files.parquet"))
    return None


def test_get_project_files(run_in_resources_directory: pytest.Function) -> Optional[NoReturn]:
    expected_files = pd.read_parquet(f"{DEFAULT_PROJECT}_files.parquet")
    received_files = pride.get_project_files(project_name=DEFAULT_PROJECT,
                                             api_versions=[DEFAULT_TEST_API])
    pd._testing.assert_frame_equal(expected_files, received_files,
                                   check_exact=True)
    return None


def store_get_raw_project_files() -> None:
    project_files = pride.get_project_files(project_name=DEFAULT_PROJECT,
                                            api_versions=[DEFAULT_TEST_API])
    project_files = mmproteo.utils.filters.filter_files_df(
        files_df=project_files, file_extensions={"raw"})
    project_files.to_parquet(os.path.join(RESOURCES_PATH, f"{DEFAULT_PROJECT}_files_raw.parquet"))


def test_list_raw_project_files(run_in_resources_directory: pytest.Function) -> Optional[NoReturn]:
    expected_files = pd.read_parquet(f"{DEFAULT_PROJECT}_files_raw.parquet")
    received_files = pride.get_project_files(project_name=DEFAULT_PROJECT,
                                             api_versions=[DEFAULT_TEST_API])
    received_files = mmproteo.utils.filters.filter_files_df(files_df=received_files, file_extensions={"raw"})
    pd._testing.assert_frame_equal(expected_files, received_files, check_exact=True)
    return None


def store_get_mgf_mzid_project_files() -> None:
    project_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    project_files = mmproteo.utils.filters.filter_files_df(files_df=project_files, file_extensions={"mgf", "mzid"})
    project_files.to_parquet(os.path.join(RESOURCES_PATH, f"{DEFAULT_PROJECT}_files_mgf_mzid.parquet"))


def test_list_mgf_mzid_files(run_in_resources_directory: pytest.Function) -> Optional[NoReturn]:
    expected_files = pd.read_parquet(f"{DEFAULT_PROJECT}_files_mgf_mzid.parquet")
    received_files = pride.get_project_files(project_name=DEFAULT_PROJECT, api_versions=[DEFAULT_TEST_API])
    received_files = mmproteo.utils.filters.filter_files_df(files_df=received_files, file_extensions={"mgf", "mzid"})
    pd._testing.assert_frame_equal(expected_files, received_files, check_exact=True)
    return None


def store_get_gz_project_files() -> None:
    project_files = pride.get_project_files(project_name=DEFAULT_PROJECT,
                                            api_versions=[DEFAULT_TEST_API])
    project_files = mmproteo.utils.filters.filter_files_df(files_df=project_files, file_extensions={"gz"})
    project_files.to_parquet(os.path.join(RESOURCES_PATH, f"{DEFAULT_PROJECT}_files_gz.parquet"))


def test_list_gz_files(run_in_resources_directory: pytest.Function) -> Optional[NoReturn]:
    expected_files = pd.read_parquet(f"{DEFAULT_PROJECT}_files_gz.parquet")
    received_files = pride.get_project_files(project_name=DEFAULT_PROJECT,
                                             api_versions=[DEFAULT_TEST_API])
    received_files = mmproteo.utils.filters.filter_files_df(files_df=received_files, file_extensions={"gz"})
    pd._testing.assert_frame_equal(expected_files, received_files, check_exact=True)
    return None


if __name__ == '__main__':
    store_get_project_info()
    store_get_project_files()
    store_get_raw_project_files()
    store_get_mgf_mzid_project_files()
    store_get_gz_project_files()
