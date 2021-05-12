import os
from typing import Dict, Set, Optional, List, Sequence

from mmproteo.utils import utils, log
from mmproteo.utils.config import Config
from mmproteo.utils.filters import AbstractFilterConditionNode, filter_files_list
from mmproteo.utils.formats import read
from mmproteo.utils.processing import ItemProcessor

_FILE_EXTRACTION_CONFIG: Dict[str, Dict[str, str]] = {
    "gz": {
        "command": 'gunzip "%s"'
    },
    "zip": {
        "command": 'unzip "%s"'
    },
}


def get_extractable_file_extensions() -> Set[str]:
    return set(_FILE_EXTRACTION_CONFIG.keys())


def get_string_of_extractable_file_extensions(extension_quote: str = Config.default_option_quote,
                                              separator: str = Config.default_option_separator) -> str:
    return utils.concat_set_of_options(options=get_extractable_file_extensions(),
                                       option_quote=extension_quote,
                                       separator=separator)


def extract_file_if_possible(filename: Optional[str],
                             skip_existing: bool = Config.default_skip_existing,
                             logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[str]:
    if filename is None:
        return None

    extracted_filename, file_ext = read.separate_extension(filename, get_extractable_file_extensions())
    if len(file_ext) == 0:
        logger.debug(f"Cannot extract file '{extracted_filename}', unknown extension")
        return None

    extraction_command = _FILE_EXTRACTION_CONFIG[file_ext]["command"] % filename

    if skip_existing and os.path.isfile(extracted_filename):
        logger.info(f'Skipping extraction, because "{extracted_filename}" already exists')
        return extracted_filename

    logger.info("Extracting file using '%s'" % extraction_command)
    return_code = os.system(extraction_command)
    if return_code == 0:
        logger.info(f"Extracted file '{extracted_filename}'")
    else:
        logger.warning(f'Failed extracting file "{filename}" (return code = {return_code})')

    return extracted_filename


class _ArchiveFileProcessor:
    def __init__(self,
                 skip_existing: bool = Config.default_skip_existing,
                 logger: log.Logger = log.DEFAULT_LOGGER):
        self.skip_existing = skip_existing
        self.logger = logger

    def __call__(self, filename: Optional[str]) -> Optional[str]:
        return extract_file_if_possible(filename=filename, skip_existing=self.skip_existing, logger=self.logger)


def extract_files(filenames: Sequence[Optional[str]],
                  skip_existing: bool = Config.default_skip_existing,
                  max_num_files: Optional[int] = None,
                  count_failed_files: bool = Config.default_count_failed_files,
                  count_skipped_files: bool = Config.default_count_skipped_files,
                  thread_count: int = Config.default_thread_count,
                  column_filter: Optional[AbstractFilterConditionNode] = None,
                  keep_null_values: bool = Config.default_keep_null_values,
                  pre_filter_files: bool = Config.default_pre_filter_files,
                  logger: log.Logger = log.DEFAULT_LOGGER) -> List[Optional[str]]:
    if pre_filter_files:
        filenames = filter_files_list(filenames=filenames,
                                      file_extensions=get_extractable_file_extensions(),
                                      max_num_files=max_num_files,
                                      column_filter=column_filter,
                                      keep_null_values=keep_null_values,
                                      drop_duplicates=not keep_null_values,
                                      sort=not keep_null_values,
                                      logger=logger)

    file_processor = _ArchiveFileProcessor(skip_existing=skip_existing, logger=logger)

    return list(ItemProcessor(items=filenames,
                              item_processor=file_processor,
                              action_name="extract",
                              subject_name="archive file",
                              max_num_items=max_num_files,
                              count_failed_items=count_failed_files,
                              count_null_results=count_skipped_files,
                              thread_count=thread_count,
                              keep_null_values=keep_null_values,
                              logger=logger).process())
