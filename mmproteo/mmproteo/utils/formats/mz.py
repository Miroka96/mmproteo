import os
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from pyteomics import mzid, mzml

from mmproteo.utils import log, utils
from mmproteo.utils.config import Config
from mmproteo.utils.filters import AbstractFilterConditionNode, filter_files_list
from mmproteo.utils.formats import read
from mmproteo.utils.processing import ItemProcessor


def merge_mzml_and_mzid_dfs(mzml_df: pd.DataFrame,
                            mzid_df: pd.DataFrame,
                            mzml_key_columns: List[str] = None,
                            mzid_key_columns: List[str] = None,
                            logger: log.Logger = log.DEFAULT_LOGGER) -> pd.DataFrame:
    if mzml_key_columns is None:
        mzml_key_columns = Config.default_mzml_key_columns
    if mzid_key_columns is None:
        mzid_key_columns = Config.default_mzid_key_columns

    input_length = min(len(mzml_df), len(mzid_df))

    logger.debug("Started merging MzML and MzID dataframes")
    merged_df = mzml_df.merge(right=mzid_df,
                              how='inner',
                              left_on=mzml_key_columns,
                              right_on=mzid_key_columns)
    output_length = len(merged_df)
    unmatched_rows = input_length - output_length
    if unmatched_rows < 0:
        if logger.is_verbose():
            unique_mzml_length = len(mzml_df.drop_duplicates(inplace=False, subset=mzml_key_columns))
            unique_mzid_length = len(mzid_df.drop_duplicates(inplace=False, subset=mzid_key_columns))
            duplicate_mzml_rows = len(mzml_df) - unique_mzml_length
            duplicate_mzid_rows = len(mzid_df) - unique_mzid_length
            logger.debug("Duplicate MzML rows = %d" % duplicate_mzml_rows)
            logger.debug("Duplicate MzID rows = %d" % duplicate_mzid_rows)
        logger.warning("The key columns were no real key columns, because duplicates were found.")

    logger.debug("Finished merging MzML and MzID dataframes and matched %d x %d -> %d rows" %
                 (len(mzml_df), len(mzid_df), output_length))
    return merged_df


def merge_mzml_and_mzid_files(mzml_filename: str,
                              mzid_filename: str,
                              mzml_key_columns: Optional[List[str]] = None,
                              mzid_key_columns: Optional[List[str]] = None,
                              logger: log.Logger = log.DEFAULT_LOGGER) -> pd.DataFrame:
    if mzml_key_columns is None:
        mzml_key_columns = Config.default_mzml_key_columns
    if mzid_key_columns is None:
        mzid_key_columns = Config.default_mzid_key_columns

    logger.debug("Started Merge: '%s' + '%s' -> dataframe" % (mzml_filename, mzid_filename))
    mzml_df = read.read(filename=mzml_filename, logger=logger)
    mzid_df = read.read(filename=mzid_filename, logger=logger)

    merged_df = merge_mzml_and_mzid_dfs(mzml_df=mzml_df,
                                        mzid_df=mzid_df,
                                        mzml_key_columns=mzml_key_columns,
                                        mzid_key_columns=mzid_key_columns,
                                        logger=logger)
    logger.debug("Finished Merge: '%s' + '%s' -> dataframe" % (mzml_filename, mzid_filename))
    return merged_df


def _process_mzml_and_mzid_to_parquet_merge_job(mzml_filename: str,
                                                mzid_filename: str,
                                                target_filename: str,
                                                current_merge_job_index: int,
                                                merge_job_count: int,
                                                skip_existing: bool = Config.default_skip_existing,
                                                mzml_key_columns: Optional[List[str]] = None,
                                                mzid_key_columns: Optional[List[str]] = None,
                                                logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[str]:
    if skip_existing and os.path.exists(target_filename):
        logger.info("Skipping Merge %d/%d: '%s' + '%s' -> '%s' already exists" %
                    (current_merge_job_index + 1, merge_job_count, mzml_filename, mzid_filename, target_filename))
        return None

    logger.info("Started Merge %d/%d: '%s' + '%s' -> '%s'" %
                (current_merge_job_index + 1, merge_job_count, mzml_filename, mzid_filename, target_filename))
    merged_df = merge_mzml_and_mzid_files(mzml_filename=mzml_filename,
                                          mzid_filename=mzid_filename,
                                          mzml_key_columns=mzml_key_columns,
                                          mzid_key_columns=mzid_key_columns,
                                          logger=logger)
    merged_df.to_parquet(path=target_filename)
    logger.info("Finished Merge %d/%d: '%s' + '%s' -> '%s'" %
                (current_merge_job_index + 1, merge_job_count, mzml_filename, mzid_filename, target_filename))

    return target_filename


class _Mz2ParquetMergeJobProcessor:
    def __init__(self,
                 merge_job_count: int,
                 skip_existing: bool = Config.default_skip_existing,
                 mzml_key_columns: Optional[List[str]] = None,
                 mzid_key_columns: Optional[List[str]] = None,
                 logger: log.Logger = log.DEFAULT_LOGGER):
        self.merge_job_count = merge_job_count
        self.skip_existing = skip_existing
        self.mzml_key_columns = mzml_key_columns
        self.mzid_key_columns = mzid_key_columns
        self.logger = logger

    def __call__(self, merge_job: Tuple[int, Tuple[str, str, str]]) -> Optional[str]:
        index, (mzml_filename, mzid_filename, target_filename) = merge_job
        return _process_mzml_and_mzid_to_parquet_merge_job(mzml_filename=mzml_filename,
                                                           mzid_filename=mzid_filename,
                                                           target_filename=target_filename,
                                                           current_merge_job_index=index,
                                                           merge_job_count=self.merge_job_count,
                                                           skip_existing=self.skip_existing,
                                                           mzml_key_columns=self.mzml_key_columns,
                                                           mzid_key_columns=self.mzid_key_columns,
                                                           logger=self.logger)


def _create_merge_jobs(filenames_and_extensions: List[Tuple[str, Tuple[str, str]]],
                       prefix_length_tolerance: int,
                       target_filename_postfix: str,
                       skip_existing: bool = Config.default_skip_existing,
                       logger: log.Logger = log.DEFAULT_LOGGER) -> List[Tuple[str, str, str]]:
    filenames_and_extensions = sorted(filenames_and_extensions)
    merge_jobs = []

    last_filename, (last_filename_prefix, last_extension) = filenames_and_extensions[0]
    for filename, (filename_prefix, extension) in filenames_and_extensions[1:]:
        if last_filename is not None and extension != last_extension:
            common_filename_prefix_length = len(os.path.commonprefix([filename_prefix, last_filename_prefix]))
            required_filename_prefix_length = min(len(filename_prefix),
                                                  len(last_filename_prefix)) - prefix_length_tolerance
            if common_filename_prefix_length >= required_filename_prefix_length:
                # found a possible merging pair
                if extension == "mzml":
                    mzml_filename = filename
                    mzid_filename = last_filename
                else:
                    mzml_filename = last_filename
                    mzid_filename = filename

                target_filename = filename[:common_filename_prefix_length] + target_filename_postfix
                if skip_existing and os.path.exists(target_filename):
                    logger.info(f"Skipping Merge: '{mzml_filename}' + '{mzid_filename}' "
                                f"-> '{target_filename}' already exists")
                else:
                    merge_jobs.append((mzml_filename, mzid_filename, target_filename))

                filename = None  # skip next iteration to prevent merging the same file with multiple others
        last_filename = filename
        last_filename_prefix = filename_prefix
        last_extension = extension

    return merge_jobs


def merge_mzml_and_mzid_files_to_parquet(filenames: List[Optional[str]],
                                         skip_existing: bool = Config.default_skip_existing,
                                         max_num_files: Optional[int] = None,
                                         count_failed_files: bool = Config.default_count_failed_files,
                                         count_skipped_files: bool = Config.default_count_skipped_files,
                                         column_filter: Optional[AbstractFilterConditionNode] = None,
                                         mzml_key_columns: Optional[List[str]] = None,
                                         mzid_key_columns: Optional[List[str]] = None,
                                         prefix_length_tolerance: int = 0,
                                         target_filename_postfix: str = Config.default_mzmlid_parquet_file_postfix,
                                         thread_count: int = Config.default_thread_count,
                                         logger: log.Logger = log.DEFAULT_LOGGER) -> List[str]:
    filenames = filter_files_list(filenames=filenames,
                                  column_filter=column_filter,
                                  sort=False,
                                  logger=logger)

    filenames_and_extensions: List[Tuple[str, Tuple[str, str]]] \
        = [(filename, read.separate_extension(filename=filename, extensions={"mzml", "mzid"}))
           for filename in filenames if filename is not None]
    filenames_and_extensions = [(filename, (file, ext)) for filename, (file, ext) in filenames_and_extensions if
                                len(ext) > 0]
    if len(filenames_and_extensions) < 2:
        logger.warning("No MzML and MzID files available for merging")
        return []

    merge_jobs = _create_merge_jobs(filenames_and_extensions=filenames_and_extensions,
                                    prefix_length_tolerance=prefix_length_tolerance,
                                    target_filename_postfix=target_filename_postfix,
                                    skip_existing=skip_existing,
                                    logger=logger)

    item_processor = _Mz2ParquetMergeJobProcessor(merge_job_count=len(merge_jobs),
                                                  skip_existing=skip_existing,
                                                  mzml_key_columns=mzml_key_columns,
                                                  mzid_key_columns=mzid_key_columns,
                                                  logger=logger)

    parquet_files = list(ItemProcessor(items=enumerate(merge_jobs),
                                       item_processor=item_processor,
                                       action_name="mzmlid-merge",
                                       subject_name="file pair",
                                       max_num_items=max_num_files,
                                       count_failed_items=count_failed_files,
                                       count_null_results=count_skipped_files,
                                       keep_null_values=False,
                                       thread_count=thread_count,
                                       logger=logger).process())
    return parquet_files


def _prepare_mzid_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    entry = entry.copy()

    try:
        items = utils.list_of_dicts_to_dict(entry["SpectrumIdentificationItem"], "rank")
        if items is not None:
            entry["SpectrumIdentificationItem"] = items
    except:
        pass

    entry = utils.flatten_dict(entry)
    return entry


def read_mzid(filename: str, logger: log.Logger = log.DEFAULT_LOGGER) -> pd.DataFrame:
    entries = read.iter_entries(mzid.read(filename), logger=logger)
    extracted_entries = [_prepare_mzid_entry(entry) for entry in entries]
    return pd.DataFrame(data=extracted_entries)


def read_mzml(filename: str, logger: log.Logger = log.DEFAULT_LOGGER) -> pd.DataFrame:
    entries = read.iter_entries(mzml.read(filename), logger=logger)
    extracted_entries = [utils.flatten_dict(entry) for entry in entries]
    return pd.DataFrame(data=extracted_entries)
