import os
from typing import Optional, List, Sequence

import pandas as pd
from mmproteo.utils import log, utils
from mmproteo.utils.config import Config
from mmproteo.utils.filters import AbstractFilterConditionNode, filter_files_list
from mmproteo.utils.formats import read
from mmproteo.utils.processing import ItemProcessor
from pyteomics import mgf


def read_mgf(filename: str, logger: log.Logger = log.DEFAULT_LOGGER) -> pd.DataFrame:
    entries = read.iter_entries(mgf.read(filename), logger=logger)
    extracted_entries = [utils.flatten_dict(entry) for entry in entries]
    return pd.DataFrame(data=extracted_entries)


def convert_mgf_file_to_parquet(filename: Optional[str],
                                skip_existing: bool = Config.default_skip_existing,
                                logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[str]:
    if filename is None:
        return None

    base_filename, file_ext = read.separate_extension(filename=filename,
                                                      extensions={"mgf"})
    if len(file_ext) == 0:
        logger.debug("Cannot convert file '%s', unknown extension" % filename)
        return None

    converted_filename = base_filename + ".parquet"

    if skip_existing and os.path.isfile(converted_filename):
        logger.info('Skipping conversion, because "%s" already exists' % converted_filename)
        return converted_filename

    try:
        df = read_mgf(filename=filename, logger=logger)
        df.to_parquet(path=converted_filename)
        logger.info("Converted file " + filename)
        return converted_filename
    except Exception as e:
        logger.warning(f'Failed converting file "{filename}" ({e})')
        return None


class _Mgf2ParquetFileProcessor:
    def __init__(self,
                 skip_existing: bool = Config.default_skip_existing,
                 logger: log.Logger = log.DEFAULT_LOGGER):
        self.skip_existing = skip_existing
        self.logger = logger

    def __call__(self, filename: Optional[str]) -> Optional[str]:
        return convert_mgf_file_to_parquet(filename=filename,
                                           skip_existing=self.skip_existing,
                                           logger=self.logger)


def convert_mgf_files_to_parquet(filenames: Sequence[Optional[str]],
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
                                      file_extensions={"mgf"},
                                      max_num_files=None,
                                      column_filter=column_filter,
                                      keep_null_values=keep_null_values,
                                      drop_duplicates=not keep_null_values,
                                      sort=not keep_null_values,
                                      logger=logger)

    file_processor = _Mgf2ParquetFileProcessor(skip_existing=skip_existing, logger=logger)

    return list(ItemProcessor(items=filenames,
                              item_processor=file_processor,
                              action_name="mgf2parquet-convert",
                              subject_name="MGF file",
                              max_num_items=max_num_files,
                              count_failed_items=count_failed_files,
                              count_null_results=count_skipped_files,
                              thread_count=thread_count,
                              keep_null_values=keep_null_values,
                              logger=logger).process())
