import gc
import os
from typing import Iterable, List, Optional, Tuple, Dict, Any, Union, Callable, Sequence

import pandas as pd
from mmproteo.utils import log, utils
from mmproteo.utils.config import Config
from mmproteo.utils.filters import AbstractFilterConditionNode, filter_files_list
from mmproteo.utils.formats import read
from mmproteo.utils.processing import ItemProcessor
from pyteomics import mzid, mzml


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


class Mz2ParquetMergeJobProcessor:
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

    def _process_mzml_and_mzid_to_parquet_merge_job(self,
                                                    mzml_filename: str,
                                                    mzid_filename: str,
                                                    target_filename: str,
                                                    current_merge_job_index: int) -> Optional[str]:
        if self.skip_existing and os.path.exists(target_filename):
            self.logger.info("Skipping Merge %d/%d: '%s' + '%s' -> '%s' already exists" %
                             (current_merge_job_index + 1, self.merge_job_count, mzml_filename, mzid_filename,
                              target_filename))
            return None

        self.logger.info("Started Merge %d/%d: '%s' + '%s' -> '%s'" %
                         (current_merge_job_index + 1, self.merge_job_count, mzml_filename, mzid_filename,
                          target_filename))
        merged_df = merge_mzml_and_mzid_files(mzml_filename=mzml_filename,
                                              mzid_filename=mzid_filename,
                                              mzml_key_columns=self.mzml_key_columns,
                                              mzid_key_columns=self.mzid_key_columns,
                                              logger=self.logger)
        merged_df.to_parquet(path=target_filename)
        self.logger.info("Finished Merge %d/%d: '%s' + '%s' -> '%s'" %
                         (current_merge_job_index + 1, self.merge_job_count, mzml_filename, mzid_filename,
                          target_filename))

        return target_filename

    def __call__(self, merge_job: Tuple[int, Tuple[str, str, str]]) -> Optional[str]:
        index, (mzml_filename, mzid_filename, target_filename) = merge_job
        return self._process_mzml_and_mzid_to_parquet_merge_job(mzml_filename=mzml_filename,
                                                                mzid_filename=mzid_filename,
                                                                target_filename=target_filename,
                                                                current_merge_job_index=index)


def _create_merge_jobs(filenames_and_extensions: List[Tuple[str, Tuple[str, str]]],
                       prefix_length_tolerance: int,
                       target_filename_postfix: str,
                       skip_existing: bool = Config.default_skip_existing,
                       logger: log.Logger = log.DEFAULT_LOGGER) -> List[Tuple[str, str, str]]:
    if len(filenames_and_extensions) < 2:
        return list()

    filenames_and_extensions = sorted(filenames_and_extensions)
    merge_jobs = []

    first_filename_and_extension: Tuple[Optional[str], Tuple[str, str]] = filenames_and_extensions[0]
    last_filename, (last_filename_prefix, last_extension) = first_filename_and_extension
    for filename, (filename_prefix, extension) in filenames_and_extensions[1:]:
        next_last_filename: Optional[str] = filename

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

                next_last_filename = None  # skip next iteration to prevent merging the same file with multiple others
        last_filename = next_last_filename
        last_filename_prefix = filename_prefix
        last_extension = extension

    return merge_jobs


def merge_mzml_and_mzid_files_to_parquet(filenames: Sequence[Optional[str]],
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

    item_processor = Mz2ParquetMergeJobProcessor(merge_job_count=len(merge_jobs),
                                                 skip_existing=skip_existing,
                                                 mzml_key_columns=mzml_key_columns,
                                                 mzid_key_columns=mzid_key_columns,
                                                 logger=logger)

    processing_results: Iterable[Optional[Any]] = ItemProcessor(items=enumerate(merge_jobs),
                                                                item_processor=item_processor.__call__,
                                                                action_name="mzmlid-merge",
                                                                subject_name="file pair",
                                                                max_num_items=max_num_files,
                                                                count_failed_items=count_failed_files,
                                                                count_null_results=count_skipped_files,
                                                                keep_null_values=False,
                                                                thread_count=thread_count,
                                                                logger=logger).process()
    parquet_files: List[str] = list(processing_results)  # type: ignore
    return parquet_files


def _prepare_mzid_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    entry = entry.copy()

    try:
        items = utils.list_of_dicts_to_dict_by_key(entry["SpectrumIdentificationItem"], "rank")
        if items is not None:
            entry["SpectrumIdentificationItem"] = items
    except Exception:
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


class FilteringProcessor:
    default_is_decoy_column_name = 'SpectrumIdentificationItem__1__PeptideEvidenceRef__0__isDecoy'
    default_fdr_column_name = 'SpectrumIdentificationItem__1__MSGFQValue'
    default_peptide_sequence_column_name = 'peptide_sequence'
    default_mz_array_column_name = 'mz_array'
    default_intensity_array_column_name = 'intensity_array'

    @staticmethod
    def get_default_output_columns() -> List[str]:
        return [
            FilteringProcessor.default_peptide_sequence_column_name,
            FilteringProcessor.default_mz_array_column_name,
            FilteringProcessor.default_intensity_array_column_name,
        ]

    def __init__(self,
                 dump_path: str,
                 fdr: float = 0.01,
                 skip_existing: bool = True,
                 is_decoy_column_name: str = default_is_decoy_column_name,
                 fdr_column_name: str = default_fdr_column_name,
                 output_columns: Optional[List[str]] = None,
                 post_processor: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
                 logger: log.Logger = log.DEFAULT_LOGGER):
        self.is_decoy_column_name = is_decoy_column_name
        self.fdr_column_name = fdr_column_name
        self.fdr = fdr
        if output_columns is None:
            self.output_columns = self.get_default_output_columns()
        else:
            self.output_columns = output_columns
        self.dump_path = dump_path.rstrip(os.path.sep)
        self.skip_existing = skip_existing
        self.post_processor = post_processor
        self.logger = logger

    def __call__(self, input_file_path: str) -> Optional[Dict[str, Union[str, int, float]]]:
        output_file_path = self.dump_path + os.path.sep + input_file_path.split(os.path.sep)[-1]
        if self.skip_existing and os.path.exists(output_file_path):
            self.logger.debug(f"Skipping filtering '{input_file_path}' -> '{output_file_path}' already exists")
            return None

        res: Dict[str, Union[str, int, float]] = {
            'input_file_path': input_file_path,
            'output_file_path': output_file_path,
        }

        df = pd.read_parquet(input_file_path)
        length = len(df)
        res['original_sequence_count'] = length

        df = df.dropna(subset=[self.is_decoy_column_name])
        new_length = len(df)
        res['NaN_decoy_count'] = length - new_length
        length = new_length

        df = df[df[self.fdr_column_name] <= self.fdr]
        new_length = len(df)
        res['above_fdr_count'] = length - new_length
        length = new_length

        decoy_counts = df[self.is_decoy_column_name].value_counts()
        left_decoys: int = decoy_counts.get(True, 0)
        left_targets: int = decoy_counts.get(False, 0)
        res['left_decoys'] = left_decoys
        res['left_targets'] = left_targets
        res['fdr'] = left_decoys / left_targets

        # filter out Decoys

        df = df[~df[self.is_decoy_column_name].astype(bool)]
        new_length = len(df)
        res['removed_decoys'] = length - new_length
        length = new_length

        df = self.post_processor(df)
        new_length = len(df)
        res['removed_by_post_processor'] = length - new_length
        length = new_length

        res['final_sequence_count'] = length

        df = df[self.output_columns]
        df.to_parquet(output_file_path)

        self.logger.info(f"Finished filtering '{input_file_path}' -> '{output_file_path}'")

        return res


def filter_files(input_file_paths: List[str],
                 output_path: str,
                 fdr: float = 0.01,
                 skip_existing: bool = True,
                 is_decoy_column_name: str = FilteringProcessor.default_is_decoy_column_name,
                 fdr_column_name: str = FilteringProcessor.default_fdr_column_name,
                 output_columns: Optional[List[str]] = None,
                 post_processor: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
                 thread_count: int = Config.default_thread_count,
                 logger: log.Logger = log.DEFAULT_LOGGER) -> List[Dict[str, Union[str, int, float]]]:
    filter_processor = FilteringProcessor(dump_path=output_path,
                                          fdr=fdr,
                                          skip_existing=skip_existing,
                                          is_decoy_column_name=is_decoy_column_name,
                                          fdr_column_name=fdr_column_name,
                                          output_columns=output_columns,
                                          post_processor=post_processor,
                                          logger=logger)
    output_files = list(ItemProcessor(
        items=input_file_paths,
        item_processor=filter_processor.__call__,
        action_name="fdr-filter",
        subject_name="mzmlid file",
        thread_count=thread_count,
        logger=logger
    ).process())
    return output_files  # type: ignore


class MzmlidFileStatsCreator:
    def __init__(
            self,
            mzmlid_file_paths: List[str],
            statistics_file_path: str,
            seq_col_name: str = FilteringProcessor.default_peptide_sequence_column_name,
            int_col_name: str = FilteringProcessor.default_intensity_array_column_name,
            logger: log.Logger = log.DEFAULT_LOGGER
    ):
        self.mzmlid_file_paths = mzmlid_file_paths
        self.file_path_count = len(mzmlid_file_paths)
        self.statistics_file_path = statistics_file_path
        self.SEQ = seq_col_name
        self.INT = int_col_name
        self.logger = logger

    def __call__(self, item: Tuple[int, str]) -> Dict[str, Any]:
        idx, path = item
        info_text = f"Processing item {idx + 1}/{self.file_path_count} '{path}'"
        if idx % 10 == 0:
            self.logger.info(info_text)
        else:
            self.logger.debug(info_text)
        df = pd.read_parquet(path)
        max_sequence_length = df[self.SEQ].str.len().max()
        max_array_length = df[self.INT].str.len().max()
        alphabet = set.union(*df[self.SEQ].apply(set))
        item_count = len(df)
        del df
        gc.collect()

        return {
            "file_path": path,
            "max_sequence_length": max_sequence_length,
            "max_array_length": max_array_length,
            "alphabet": alphabet,
            "item_count": item_count
        }

    def process(self, **kwargs: Dict[str, Any]) -> pd.DataFrame:
        if os.path.exists(self.statistics_file_path):
            file_stats = pd.read_parquet(self.statistics_file_path)
            file_stats.alphabet = file_stats.alphabet.apply(set)
            self.logger.info(f"loaded previous statistics file '{self.statistics_file_path}'")
            return file_stats
        else:
            file_stats = pd.DataFrame(
                ItemProcessor(
                    items=enumerate(self.mzmlid_file_paths),
                    item_processor=self.__call__,
                    action_name="analyse",
                    subject_name="mzmlid file",
                    logger=self.logger,
                    **kwargs  # type: ignore
                ).process()
            )

            file_stats_writable = file_stats.copy()
            file_stats_writable.alphabet = file_stats_writable.alphabet.apply(list)  # cannot store sets
            file_stats_writable.to_parquet(self.statistics_file_path)

            self.logger.info(f"saved statistics to '{self.statistics_file_path}' file")

            return file_stats
