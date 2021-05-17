import os

from typing import Set, Optional, NoReturn

import pytest

from mmproteo.utils import utils, commands, config, log
from .utils.defaults import MZML_FILE, MZID_FILE, MZMLID_FILE, \
    DEFAULT_PROJECT_FILE_COUNT, DEFAULT_PROJECT, MZID_GZ_FILE, MZML_GZ_FILE, \
    RAW_FILE, MGF_FILE, MGF_PARQUET_FILE

pytest.skip("skipping long-running tests", allow_module_level=True)


def get_files_in_directory(directory_name: str) -> Set[str]:
    directory, directories, files = next(os.walk(directory_name))
    return set(files)


def _run_mz_pipeline_combined(conf: config.Config, logger: log.Logger) \
        -> Optional[NoReturn]:
    logger.fail_early = True
    conf.commands = ["list", "download", "extract", "mz2parquet"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)

    assert os.path.isfile(MZML_FILE), \
        "the mzml file should have been downloaded and extracted"
    assert os.path.isfile(MZID_FILE), \
        "the mzid file should have been downloaded and extracted"
    assert os.path.isfile(MZMLID_FILE), \
        "the mzml file and the mzid file should have been merged"

    assert get_files_in_directory(".") == {MZML_FILE, MZID_FILE, MZMLID_FILE}, \
        "there should only be the mzML, mzid, and parquet file"

    assert conf._project_files is not None, "the project files should have been cached"
    assert len(conf._project_files) == DEFAULT_PROJECT_FILE_COUNT, \
        f"there should be {DEFAULT_PROJECT_FILE_COUNT} cached project files"

    assert conf._processed_files is not None, "processed files should be " \
                                              "cached by now"

    assert conf.default_downloaded_files_column in conf._processed_files, \
        "there should be references to downloaded files"
    assert len(conf._processed_files[
                   conf.default_downloaded_files_column].dropna()) == conf.max_num_files, \
        f"there should be exactly {conf.max_num_files} referenced downloaded files"

    assert conf.default_extracted_files_column in conf._processed_files, \
        "there should be references to extracted files"
    assert len(conf._processed_files[
                   conf.default_extracted_files_column].dropna()) == conf.max_num_files, \
        f"there should be exactly {conf.max_num_files} referenced extracted files"

    assert conf.default_mzmlid_parquet_files_column in conf._processed_files, \
        f"there should be references to converted '*{conf.default_mzmlid_parquet_file_postfix}' files"
    assert len(conf._processed_files[
                   conf.default_mzmlid_parquet_files_column].dropna()) == conf.max_num_files / 2, \
        f"there should be exactly {conf.max_num_files / 2} referenced converted " \
        f"'*{conf.default_mzmlid_parquet_file_postfix}' file"

    conf.clear_cache()

    return None


def _run_mz_pipeline_in_single_steps_with_full_cache(conf: config.Config,
                                                     logger: log.Logger) \
        -> Optional[NoReturn]:
    logger.fail_early = False
    conf.commands = ["download"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)

    downloaded_files = utils.merge_column_values(conf._processed_files, [
        conf.default_downloaded_files_column])
    assert len(
        downloaded_files) == 0, "there should be no downloaded files in the second iteration"

    conf.clear_cache()
    conf.commands = ["extract"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)

    extracted_files = utils.merge_column_values(conf._processed_files, [
        conf.default_extracted_files_column])
    assert len(
        extracted_files) == 0, "there should be no extracted files in the second iteration"

    conf.clear_cache()
    conf.commands = ["mz2parquet"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)

    downloaded_and_extracted_files = utils.merge_column_values(
        conf._processed_files,
        [conf.default_mzmlid_parquet_files_column])

    assert len(
        downloaded_and_extracted_files) == 0, "there should be no converted files in the second iteration"

    assert get_files_in_directory(".") == {MZML_FILE, MZID_FILE, MZMLID_FILE}, \
        "there should only be the mzML, mzid, and parquet file"

    return None


def test_mz_pipeline_single_threaded(run_in_temp_directory: pytest.Function) \
        -> Optional[NoReturn]:
    conf = config.Config()
    conf.pride_project = DEFAULT_PROJECT
    conf.valid_file_extensions = ["mzid", "mzml"]
    conf.max_num_files = 2
    conf.shown_columns = ["fileName", "fileSizeBytes"]
    conf.thread_count = 1

    logger = log.TestLogger(terminate_process=False,
                            verbose=True)
    conf.check(logger)

    _run_mz_pipeline_combined(conf=conf, logger=logger)
    _run_mz_pipeline_in_single_steps_with_full_cache(conf=conf, logger=logger)

    return None


def test_mz_pipeline_in_parallel(run_in_temp_directory: pytest.Function) \
        -> Optional[NoReturn]:
    conf = config.Config()
    conf.pride_project = DEFAULT_PROJECT
    conf.valid_file_extensions = ["mzid", "mzml"]
    conf.max_num_files = 2
    conf.shown_columns = ["fileName", "fileSizeBytes"]
    conf.thread_count = 2

    logger = log.TestLogger(terminate_process=False,
                            verbose=True)
    conf.check(logger)

    _run_mz_pipeline_combined(conf=conf, logger=logger)
    _run_mz_pipeline_in_single_steps_with_full_cache(conf=conf, logger=logger)

    return None


def test_mz_pipeline_in_single_steps(run_in_temp_directory: pytest.Function) \
        -> Optional[NoReturn]:
    conf = config.Config()
    conf.pride_project = DEFAULT_PROJECT
    conf.max_num_files = 1

    logger = log.TestLogger(fail_early=True,
                            terminate_process=False,
                            verbose=True)
    conf.check(logger=logger)

    conf.commands = ["download"]
    conf.valid_file_extensions = ["mzml"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)
    conf.clear_cache()

    assert os.path.isfile(MZML_GZ_FILE), \
        "the mzML.gz file should have been downloaded"

    conf.commands = ["download"]
    conf.valid_file_extensions = ["mzid"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)
    conf.clear_cache()

    assert os.path.isfile(MZID_GZ_FILE), \
        "the mzid.gz file should have been downloaded"

    conf.commands = ["extract"]
    conf.valid_file_extensions = ["mzml"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)
    conf.clear_cache()

    assert os.path.isfile(MZML_FILE), \
        "the mzML file should have been extracted"

    assert not os.path.isfile(MZML_GZ_FILE), \
        "the mzML.gz file should have been deleted by gzip"

    conf.commands = ["extract"]
    conf.valid_file_extensions = ["mzid"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)
    conf.clear_cache()

    assert os.path.isfile(MZID_FILE), \
        "the mzid file should have been extracted"

    assert not os.path.isfile(MZID_GZ_FILE), \
        "the mzid.gz file should have been deleted by gzip"

    conf.commands = ["mz2parquet"]
    conf.valid_file_extensions = []
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)
    conf.clear_cache()

    assert os.path.isfile(MZMLID_FILE), \
        "the mzml file and the mzid file should have been merged"

    assert get_files_in_directory(".") == {MZML_FILE, MZID_FILE, MZMLID_FILE}, \
        "there should only be the mzML, mzid, and parquet file"

    return None


def _run_raw_mgf_pipeline_combined(conf: config.Config, logger: log.Logger) \
        -> Optional[NoReturn]:
    logger.fail_early = True
    conf.commands = ["download", "convertraw", "mgf2parquet"]
    commands.DISPATCHER.dispatch_commands(config=conf,
                                          catch_validation_warnings=False,
                                          catch_run_warnings=False,
                                          logger=logger)

    assert os.path.isfile(RAW_FILE), \
        "the raw file should have been downloaded"
    assert os.path.isfile(MGF_FILE), \
        "the raw file should have been converted to mgf format"
    assert os.path.isfile(MGF_PARQUET_FILE), \
        "the mgf file should have been converted to parquet format"

    assert conf._processed_files is not None, "processed files should be " \
                                              "cached by now"

    assert conf.default_downloaded_files_column in conf._processed_files, \
        "there should be references to downloaded files"
    assert len(conf._processed_files[
                   conf.default_downloaded_files_column].dropna()) == conf.max_num_files, \
        f"there should be exactly {conf.max_num_files} referenced downloaded file"

    assert conf.default_converted_raw_files_column in conf._processed_files, \
        "there should be references to converted raw files"
    assert len(conf._processed_files[
                   conf.default_converted_raw_files_column].dropna()) == conf.max_num_files, \
        f"there should be exactly {conf.max_num_files} referenced converted raw file"

    assert conf.default_mgf_parquet_files_column in conf._processed_files, \
           "there should be references to converted mgf files in parquet format"
    assert len(conf._processed_files[
                   conf.default_mgf_parquet_files_column].dropna()) == conf.max_num_files, \
        f"there should be exactly {conf.max_num_files} referenced converted mgf files in parquet format"

    if conf.max_num_files == 1:
        assert get_files_in_directory(".") == {RAW_FILE, MGF_FILE,
                                               MGF_PARQUET_FILE}, \
            "there should only be the mzML, mzid, and parquet file"

    return None


def test_raw_mgf_pipeline_single_threaded(
        run_in_temp_directory: pytest.Function) -> Optional[NoReturn]:
    conf = config.Config()
    conf.pride_project = DEFAULT_PROJECT
    conf.valid_file_extensions = ["raw"]
    conf.max_num_files = 1
    conf.thread_count = 1

    logger = log.TestLogger(terminate_process=False,
                            verbose=True)
    conf.check(logger)

    return _run_raw_mgf_pipeline_combined(conf=conf, logger=logger)


def test_raw_mgf_pipeline_parallelized(
        run_in_temp_directory: pytest.Function) -> Optional[NoReturn]:
    conf = config.Config()
    conf.pride_project = DEFAULT_PROJECT
    conf.valid_file_extensions = ["raw"]
    conf.max_num_files = 2
    conf.thread_count = 2

    logger = log.TestLogger(terminate_process=False,
                            verbose=True)
    conf.check(logger)

    return _run_raw_mgf_pipeline_combined(conf=conf, logger=logger)
