import json
from typing import List, Optional, Union, Dict, Any, Callable
from urllib.parse import quote

import pandas as pd
from mmproteo.utils import log, utils

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


def urlencode_df_columns(files_df: pd.DataFrame, columns: List[str], inplace: bool = True) -> pd.DataFrame:
    if not inplace:
        files_df = files_df.copy()

    for col in columns:
        if col in files_df.columns:
            files_df[col] = files_df[col].map(lambda s: quote(s, safe='/:'))

    return files_df


def print_df(df: Optional[pd.DataFrame],
             max_num_files: Optional[int] = None,
             shown_columns: Optional[List[str]] = None,
             urlencode_columns: List[str] = None,
             reset_index: bool = True,
             logger: log.Logger = log.DEFAULT_LOGGER) -> None:
    if df is None or len(df) == 0:
        logger.debug("There are no entries to be shown")
        return

    df = df.copy()

    if max_num_files is not None and max_num_files != 0 and max_num_files < len(df):
        df = df[:max_num_files]
        logger.info("Showing only the top %d entries because of the max_num_files parameter" % max_num_files)
    if shown_columns is not None and shown_columns != []:
        columns = [col for col in shown_columns if col in df.columns]
        columns = utils.deduplicate_list(columns)
        df = df[columns]
        logger.info("Limiting the shown columns according to the shown_columns parameter")

    if urlencode_columns is not None:
        df = urlencode_df_columns(df, urlencode_columns, inplace=False)

    if reset_index:
        df = df.reset_index(drop=True)

    try:
        print(df)
    except BrokenPipeError:
        pass


def pretty_print_json(item: Optional[Union[Dict, List]]) -> str:
    if item is None:
        return ""
    try:
        return json.dumps(item, indent=4)
    except TypeError as e:
        print(e)
        return str(item)


def print_list_length_in_dict(
        dic: Dict[str, List[Any]],
        print_func: Callable[[str], None] = print,
        print_example: bool = True
) -> None:
    for key, list_value in dic.items():
        print_func(f"#{key} = {len(list_value)}")
        if print_example and len(list_value) > 0:
            print_func(f"e.g.: {list_value[0]}")
