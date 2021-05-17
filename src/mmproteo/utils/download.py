import json
import os
from typing import List, NoReturn, Optional, Union, Tuple, Any, Dict

import requests
import wget
from mmproteo.utils import log, utils
from mmproteo.utils.config import Config
from mmproteo.utils.formats import archives, read
from mmproteo.utils.processing import ItemProcessor
from mmproteo.utils.visualization import pretty_print_json
from requests import Response


def download_file(download_url: str, skip_existing: bool = Config.default_skip_existing) \
        -> Tuple[Optional[str], Optional[str]]:
    filename = download_url.split("/")[-1]
    downloaded_file_name: Optional[str] = None

    found_downloaded_file: bool = False
    found_extracted_file: bool = False

    skip_reason: Optional[str] = None

    if len(filename) > 0 and skip_existing:
        if os.path.isfile(filename):
            found_downloaded_file = True
            downloaded_file_name = filename
            skip_reason = 'file "%s" already exists' % downloaded_file_name

        extracted_file_name, extension = read.separate_extension(filename, archives.get_extractable_file_extensions())
        file_is_extractable = len(extension) > 0

        if file_is_extractable:
            # gunzip deletes the archive file after extraction, other programmes keep it
            if os.path.isfile(extracted_file_name):
                found_extracted_file = True
                skip_reason = 'extracted file "%s" already exists' % extracted_file_name

    if not (skip_existing and (found_downloaded_file or found_extracted_file)):
        downloaded_file_name = wget.download(download_url)  # might raise an EXCEPTION

    return downloaded_file_name, skip_reason


class _DownloadUrlProcessor:
    def __init__(self,
                 download_url_count: int,
                 skip_existing: bool = Config.default_skip_existing,
                 logger: log.Logger = log.DEFAULT_LOGGER):
        self.download_url_count = download_url_count
        self.skip_existing = skip_existing
        self.logger = logger

    def __call__(self, indexed_url: Tuple[int, str]) -> Union[Optional[str], NoReturn]:
        current_download_index, url = indexed_url
        return handle_file_download(download_url=url,
                                    current_download_index=current_download_index,
                                    download_count=self.download_url_count,
                                    skip_existing=self.skip_existing,
                                    logger=self.logger)


def handle_file_download(download_url: str,
                         current_download_index: int,
                         download_count: int,
                         skip_existing: bool = Config.default_skip_existing,
                         logger: log.Logger = log.DEFAULT_LOGGER) -> Union[Optional[str], NoReturn]:
    logger.info(f"Downloading file {current_download_index + 1}/{download_count}: {download_url}")

    try:
        downloaded_file_name, skip_reason = download_file(download_url, skip_existing)
    except Exception as e:
        logger.info(f'Failed to download file {current_download_index + 1}/{download_count} ("{download_url}") '
                    f'because of "{e}"')
        raise

    if skip_reason is not None:
        logger.info('Skipped download, because ' + skip_reason)
        return None
    else:
        logger.info(f'Downloaded file {current_download_index + 1}/{download_count}: "{download_url}"')
        return downloaded_file_name


def download_files(download_urls: List[str],
                   skip_existing: bool = Config.default_skip_existing,
                   max_num_files: Optional[int] = None,
                   count_skipped_files: bool = Config.default_count_skipped_files,
                   count_failed_files: bool = Config.default_count_failed_files,
                   keep_null_values: bool = Config.default_keep_null_values,
                   thread_count: int = Config.default_thread_count,
                   logger: log.Logger = log.DEFAULT_LOGGER) -> List[Optional[str]]:
    download_count = len(download_urls)
    logger.info(f"Downloading {download_count} file{utils.get_plural_s(download_count)}")

    download_url_processor = _DownloadUrlProcessor(download_url_count=download_count,
                                                   skip_existing=skip_existing,
                                                   logger=logger)
    item_processor = ItemProcessor(items=enumerate(download_urls),
                                   item_processor=download_url_processor,
                                   action_name="download",
                                   subject_name="URL",
                                   keep_null_values=keep_null_values,
                                   max_num_items=max_num_files,
                                   count_null_results=count_skipped_files,
                                   count_failed_items=count_failed_files,
                                   thread_count=thread_count,
                                   logger=logger)
    downloaded_files_names: List[Optional[str]] = list(item_processor.process())
    return downloaded_files_names


class AbstractDownloader:
    def __init__(self, logger: log.Logger = log.DEFAULT_LOGGER):
        self.logger = logger

    @staticmethod
    # HTTP 204 - No Content
    def _handle_204_response(logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[NoReturn]:
        logger.warning("Repository does not exist")
        return None

    @staticmethod
    # HTTP 401 - Unauthorized
    def _handle_401_response(response_dict: dict, logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[NoReturn]:
        message = response_dict.get('message', "?")
        developer_message = response_dict.get('developerMessage', "?")
        more_info_url = response_dict.get('moreInfoUrl', "?")
        logger.warning("%s (%s) -> %s" % (message, developer_message, more_info_url))
        return None

    @staticmethod
    def _handle_unknown_response(status_code: int, response_dict: dict,
                                 logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[NoReturn]:
        logger.debug(pretty_print_json(response_dict))
        logger.warning("Received unknown response code %d or content" % status_code)
        return None

    def _handle_non_200_response_codes(self, response: Optional[Response],
                                       logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[NoReturn]:
        if response is None:
            return None
        if response.status_code == 204:
            self._handle_204_response(logger=logger)
            return None
        try:
            response_dict = json.loads(response.text)
        except json.JSONDecodeError:
            logger.warning("Received unknown non-JSON response with response code %d" % response.status_code)
            logger.debug("Response text: '%s'" % response.text)
            return None
        if response.status_code == 401:
            self._handle_401_response(response_dict, logger)
            return None
        self._handle_unknown_response(response.status_code, response_dict, logger)
        return None

    def request_json_object(self,
                            url: str,
                            subject_name: str,
                            logger: log.Logger = log.DEFAULT_LOGGER) \
            -> Union[Dict[str, Any], List[Any], NoReturn, None]:
        logger.info(f"Requesting {subject_name} from {url}")
        response = requests.get(url)
        logger.debug(f"Received response from {url} with length of "
                     f"{len(response.text)} bytes and "
                     f"status code {response.status_code}")

        if response.status_code == 200:
            response_object: Union[Dict[str, Any], List[Any]] = json.loads(response.text)
            return response_object
        else:
            self._handle_non_200_response_codes(response, logger)
            return None
