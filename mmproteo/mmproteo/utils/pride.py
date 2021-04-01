import json
import os
from typing import Dict, Iterable, List, Optional, Union, NoReturn

import pandas as pd

import mmproteo.utils.filters
from mmproteo.utils import download as dl, log, utils
from mmproteo.utils.config import Config
from mmproteo.utils.download import AbstractDownloader
from mmproteo.utils.visualization import pretty_print_json


class AbstractPrideApi(AbstractDownloader):
    LIST_PROJECT_FILES_URL = None
    GET_PROJECT_SUMMARY_URL = None

    def __init__(self, version: str, logger: log.Logger = log.DEFAULT_LOGGER):
        super().__init__(logger)
        self.version = version

    def get_project_summary(self, project_name: str) -> Optional[dict]:
        project_summary_link = self.GET_PROJECT_SUMMARY_URL % project_name
        return self.request_json(project_summary_link, "project summary using API version " + self.version,
                                 self.logger)

    def get_project_files(self, project_name: str):
        raise NotImplementedError


class PrideApiV1(AbstractPrideApi):
    LIST_PROJECT_FILES_URL = "https://www.ebi.ac.uk/pride/ws/archive/file/list/project/%s"
    GET_PROJECT_SUMMARY_URL = "https://www.ebi.ac.uk/pride/ws/archive/project/%s"

    def get_project_files(self, project_name: str) -> Optional[pd.DataFrame]:
        project_files_link = self.LIST_PROJECT_FILES_URL % project_name
        response_dict = self.request_json(url=project_files_link,
                                          subject_name="list of project files using API version " + self.version,
                                          logger=self.logger)
        if response_dict is None:
            return None
        try:
            files_df = pd.DataFrame(pd.json_normalize(response_dict['list']))
            return files_df
        except Exception:
            return None


class PrideApiV2(AbstractPrideApi):
    LIST_PROJECT_FILES_URL = "https://www.ebi.ac.uk/pride/ws/archive/v2/files/byProject?accession=%s"
    GET_PROJECT_SUMMARY_URL = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/%s"

    @staticmethod
    def _get_ftp_or_http_file_location(file_location_entries: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        compatible_file_location_entries = [entry for entry in file_location_entries
                                            if 'value' in entry
                                            and entry['value'].lower().startswith(('ftp://', 'http://', 'https://'))]
        if len(compatible_file_location_entries) > 0:
            return compatible_file_location_entries[0]
        return None

    @staticmethod
    def _format_file_entry(file_entry: dict) -> dict:
        if 'publicFileLocations' in file_entry:
            try:
                file_entry['publicFileLocations'] = PrideApiV2._get_ftp_or_http_file_location(
                    file_entry['publicFileLocations'])
                file_entry['publicFileLocation'] = file_entry.pop('publicFileLocations')  # rename to singular
            except Exception:
                pass
        return file_entry

    def get_project_files(self,
                          project_name: str,
                          download_link_column: str = Config.default_download_link_column) -> Optional[pd.DataFrame]:
        project_files_link = self.LIST_PROJECT_FILES_URL % project_name
        response_dict_list = self.request_json(url=project_files_link,
                                               subject_name="list of project files using API version " + self.version,
                                               logger=self.logger)
        if response_dict_list is None:
            return None
        response_dict_list = [self._format_file_entry(response_dict) for response_dict in response_dict_list]

        try:
            files_df = pd.DataFrame(pd.json_normalize(response_dict_list))
        except Exception:
            return None

        files_df = files_df.rename(columns={"publicFileLocation.value": download_link_column})
        return files_df


PRIDE_APIS = {
    "1": PrideApiV1,
    "2": PrideApiV2,
}

DEFAULT_PRIDE_API_VERSIONS = ["2", "1"]


def get_pride_api_versions() -> List[str]:
    return sorted(PRIDE_APIS.keys())


def get_string_of_pride_api_versions(extension_quote: str = '"', separator: str = ", ") -> str:
    return utils.concat_set_of_options(options=DEFAULT_PRIDE_API_VERSIONS,
                                       option_quote=extension_quote,
                                       separator=separator)


def _query_project_summary(project_name: str,
                           api_versions: List[str] = None,
                           logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[dict]:
    """Get the project as a json and return it as a dataframe"""
    if api_versions is None or len(api_versions) == 0:
        api_versions = DEFAULT_PRIDE_API_VERSIONS

    for api_version in api_versions:
        api = PRIDE_APIS[api_version](api_version, logger)
        response_dict = api.get_project_summary(project_name)
        if response_dict is not None:
            logger.info("Received project summary using API version %s for project \"%s\"" %
                        (api_version, project_name))
            return response_dict

    logger.warning("Could not get project summary.")
    return None


def get_project_info(project_name: str,
                     api_versions: List[str] = None,
                     logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[str]:
    summary_dict = _query_project_summary(project_name=project_name, api_versions=api_versions, logger=logger)
    if summary_dict is None:
        return None
    return pretty_print_json(summary_dict)


def get_project_files(project_name: str,
                      api_versions: List[str] = None,
                      logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[pd.DataFrame]:
    """Get the project as a json and return it as a dataframe"""
    if api_versions is None or len(api_versions) == 0:
        api_versions = DEFAULT_PRIDE_API_VERSIONS

    for api_version in api_versions:
        api = PRIDE_APIS[api_version](api_version, logger)
        files_df = api.get_project_files(project_name)
        if files_df is not None:
            logger.info("Received project file list using API version %s with %d files for project \"%s\"" %
                        (api_version, len(files_df), project_name))
            return files_df

    logger.warning("Could not get list of project files.")
    return None


def download(project_name: Optional[str] = None,
             project_files: Optional[pd.DataFrame] = None,
             valid_file_extensions: Optional[Iterable[str]] = None,
             max_num_files: Optional[int] = None,
             column_filter: Optional[mmproteo.utils.filters.AbstractFilterConditionNode] = None,
             download_dir: str = Config.default_storage_dir,
             skip_existing: bool = Config.default_skip_existing,
             count_failed_files: bool = Config.default_count_failed_files,
             count_skipped_files: bool = Config.default_count_skipped_files,
             file_name_column: str = Config.default_file_name_column,
             download_link_column: str = Config.default_download_link_column,
             downloaded_files_column: str = Config.default_downloaded_files_column,
             api_versions: List[str] = None,
             thread_count: int = Config.default_thread_count,
             logger: log.Logger = log.DEFAULT_LOGGER) \
        -> Union[Optional[pd.DataFrame], NoReturn]:
    logger.assert_true(condition=(project_name is not None or project_files is not None),
                       error_msg="either project_name or project_files are required")

    if project_files is None:
        project_files = get_project_files(project_name=project_name,
                                          api_versions=api_versions,
                                          logger=logger)
        if project_files is None:
            return None

    filtered_files = mmproteo.utils.filters.filter_files_df(files_df=project_files,
                                                            file_name_column=file_name_column,
                                                            file_extensions=valid_file_extensions,
                                                            max_num_files=None,
                                                            column_filter=column_filter,
                                                            sort=True,
                                                            logger=logger)
    logger.assert_true(download_link_column in filtered_files.columns,
                       f"Could not find column '{download_link_column}' in filtered_files dataframe")

    initial_directory = os.getcwd()
    os.chdir(download_dir)
    downloaded_files = dl.download_files(download_urls=filtered_files[download_link_column],
                                         skip_existing=skip_existing,
                                         count_failed_files=count_failed_files,
                                         count_skipped_files=count_skipped_files,
                                         max_num_files=max_num_files,
                                         keep_null_values=True,
                                         thread_count=thread_count,
                                         logger=logger)

    filtered_files = filtered_files.head(len(downloaded_files))
    filtered_files[downloaded_files_column] = downloaded_files
    os.chdir(initial_directory)

    return filtered_files
