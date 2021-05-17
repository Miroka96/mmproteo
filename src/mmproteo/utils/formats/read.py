import os
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple

import pandas as pd
from mmproteo.utils import log, visualization
from mmproteo.utils.formats.mgf import read_mgf
from mmproteo.utils.formats.mz import read_mzid, read_mzml
from pyteomics.mgf import MGFBase
from pyteomics.mzid import MzIdentML
from pyteomics.mzml import MzML


def iter_entries(iterator: Union[MGFBase, MzIdentML, MzML], logger: log.Logger = log.DEFAULT_LOGGER) \
        -> List[Dict[str, Any]]:
    logger.debug("Iterator type = " + str(type(iterator)))
    entries = list(iterator)
    logger.debug("Length: %d" % len(entries))
    if len(entries) > 0:
        logger.debug("Example:\n" + visualization.pretty_print_json(entries[0]))
    return entries


def read_parquet(filename: str, logger: log.Logger = log.DEFAULT_LOGGER) -> pd.DataFrame:
    return pd.read_parquet(filename)


_FILE_READING_CONFIG: Dict[str, Callable[[str, log.Logger], pd.DataFrame]] = {
    "mgf": read_mgf,
    "mzid": read_mzid,
    "mzml": read_mzml,
    "parquet": read_parquet,
}


def get_readable_file_extensions() -> Set[str]:
    return set(_FILE_READING_CONFIG.keys())


def read(filename: str, filename_col: Optional[str] = "%s_filename", logger: log.Logger = log.DEFAULT_LOGGER) \
        -> pd.DataFrame:
    _, ext = separate_extension(filename=filename, extensions=get_readable_file_extensions())

    if len(ext) > 0:
        logger.debug("Started reading %s file '%s'" % (ext, filename))
        df = _FILE_READING_CONFIG[ext](filename, logger)
        logger.info("Finished reading %s file '%s'" % (ext, filename))
    else:
        raise NotImplementedError
    if filename_col is not None:
        if "%s" in filename_col:
            col = filename_col % ext
        else:
            col = filename_col
        if col in df.columns:
            logger.warning("'%s' already exists in DataFrame columns. Please choose a different column name.")
        else:
            df[col] = filename.split(os.sep)[-1]
    return df


def separate_extension(filename: str, extensions: Set[str]) -> Tuple[str, str]:
    lower_filename = filename.lower()
    longest_extension = ""
    for ext in extensions:
        if lower_filename.endswith("." + ext) and len(longest_extension) < len(ext):
            longest_extension = ext

    if len(longest_extension) > 0:
        base_filename = filename[:len(filename) - len(longest_extension) - len(".")]
        return base_filename, longest_extension
    else:
        return filename, ""
