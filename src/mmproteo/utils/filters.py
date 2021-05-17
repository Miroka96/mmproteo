import argparse
import re
from typing import Callable, Collection, Iterable, List, NoReturn, Optional, Union, Sequence

import pandas as pd

from mmproteo.utils import log, utils
from mmproteo.utils.config import Config


class AbstractFilterConditionNode:
    def __call__(self, row: pd.Series) -> Optional[bool]:
        raise NotImplementedError


class AndFilterConditionNode(AbstractFilterConditionNode):
    def __init__(self, conditions: List[AbstractFilterConditionNode]):
        super(AndFilterConditionNode, self).__init__()
        self.conditions = conditions

    def __call__(self, row: pd.Series) -> Optional[bool]:
        res = None
        for condition in self.conditions:
            inner_res = condition(row=row)
            if inner_res is not None:
                if not inner_res:
                    return False
                else:
                    res = True
        return res


class OrFilterConditionNode(AbstractFilterConditionNode):
    def __init__(self, conditions: List[AbstractFilterConditionNode]):
        super(OrFilterConditionNode, self).__init__()
        self.conditions = conditions

    def __call__(self, row: pd.Series) -> Optional[bool]:
        res = None
        for condition in self.conditions:
            inner_res = condition(row=row)
            if inner_res is not None:
                if inner_res:
                    return True
                else:
                    res = False
        return res


class NotFilterConditionNode(AbstractFilterConditionNode):
    def __init__(self, condition: AbstractFilterConditionNode):
        super(NotFilterConditionNode, self).__init__()
        self.condition = condition

    def __call__(self, row: pd.Series) -> Optional[bool]:
        res = self.condition(row=row)
        if res is None:
            return None
        return not res


class ColumnRegexFilterConditionNode(AbstractFilterConditionNode):
    def __init__(self, column_name: str, value_regex: str):
        super(ColumnRegexFilterConditionNode, self).__init__()
        self.column_name = column_name
        self.value_regex = re.compile(value_regex)

    def __call__(self, row: pd.Series) -> Optional[bool]:
        if self.column_name in row.index:
            return bool(self.value_regex.fullmatch(str(row[self.column_name])))
        return None


class NoneFilterConditionNode(AbstractFilterConditionNode):
    def __init__(self, condition: AbstractFilterConditionNode, none_value: bool = True):
        super(NoneFilterConditionNode, self).__init__()
        self.condition = condition
        self.none_value = none_value

    def __call__(self, row: pd.Series) -> bool:
        res = self.condition(row=row)
        if res is None:
            res = self.none_value
        return res


def create_or_filter_from_str(or_filter_str: str) -> Union[OrFilterConditionNode, NoReturn]:
    conditions: List[AbstractFilterConditionNode] = []

    column_condition_strings = or_filter_str.split(Config.default_filter_or_separator)
    for column_condition_string in column_condition_strings:
        parts = re.split(pattern=Config.default_filter_separator_regex,
                         string=column_condition_string,
                         maxsplit=1)
        if len(parts) != 2 or len(parts[0]) == 0 or len(parts[1]) == 0:
            raise argparse.ArgumentTypeError(f"'{column_condition_string}' does not match the column filter pattern "
                                             f"'.+{Config.default_filter_separator_regex}.+'")
        column_name = parts[0]
        value_regex = parts[1]
        negation = column_condition_string[len(parts[0])] == "!"

        column_condition: AbstractFilterConditionNode \
            = ColumnRegexFilterConditionNode(column_name=column_name,
                                             value_regex=value_regex)

        if negation:
            column_condition = NotFilterConditionNode(condition=column_condition)

        conditions.append(column_condition)

    or_condition = OrFilterConditionNode(conditions=conditions)
    return or_condition


def create_file_extension_filter(required_file_extensions: Iterable[str],
                                 optional_file_extensions: Optional[Iterable[str]] = None) \
        -> Callable[[Optional[str]], bool]:
    if optional_file_extensions is None:
        file_extensions = set(required_file_extensions)
    else:
        file_extensions = {required_extension + "." + optional_extension
                           for required_extension in required_file_extensions
                           for optional_extension in optional_file_extensions}
        file_extensions.update(required_file_extensions)  # add all

    def filter_file_extension(filename: Optional[str]) -> bool:
        if filename is None:
            return True
        return filename.lower().endswith(tuple(file_extensions))

    return filter_file_extension


def filter_files_df(files_df: Optional[pd.DataFrame],
                    file_name_column: str = Config.default_file_name_column,
                    file_extensions: Optional[Collection[str]] = None,
                    column_filter: Optional[AbstractFilterConditionNode] = None,
                    max_num_files: Optional[int] = None,
                    sort: bool = Config.default_filter_sort,
                    logger: log.Logger = log.DEFAULT_LOGGER) -> Optional[pd.DataFrame]:
    if files_df is None:
        return None

    if file_extensions is None or len(file_extensions) == 0:
        logger.debug("Skipping file extension filtering")
    else:
        logger.assert_true(file_name_column in files_df.columns, "Could not find '%s' column in files_df columns" %
                           file_name_column)
        required_file_extensions = file_extensions

        from mmproteo.utils.formats.archives import get_extractable_file_extensions
        optional_file_extensions = get_extractable_file_extensions()
        optional_file_extensions -= set(required_file_extensions)

        required_file_extensions_list_str = "\", \"".join(sorted(required_file_extensions))
        optional_file_extensions_list_str = "\", \"".join(sorted(optional_file_extensions))
        if len(optional_file_extensions_list_str) > 0:
            optional_file_extensions_list_str = " and the following optional file extensions " \
                                                f"[\"{optional_file_extensions_list_str}\"]"
        logger.info("Filtering files based on the following required file extensions "
                    f"[\"{required_file_extensions_list_str}\"]" + optional_file_extensions_list_str)

        file_extension_filter = create_file_extension_filter(required_file_extensions, optional_file_extensions)
        files_df = files_df[files_df[file_name_column].apply(file_extension_filter)]

        logger.debug(f"File extension filtering resulted in {len(files_df)} valid file "
                     f"name{utils.get_plural_s(len(files_df))}")

        if len(files_df) == 0:
            return files_df

    if column_filter is not None:
        column_filter = NoneFilterConditionNode(
            condition=column_filter, none_value=True)
        files_df = files_df[files_df.apply(func=column_filter.__call__, axis=1)]  # type: ignore

        logger.debug(f"File attribute filtering resulted in {len(files_df)} valid file "
                     f"name{utils.get_plural_s(len(files_df))}")

        if len(files_df) == 0:
            return files_df

    if sort:
        # sort, such that files with same prefixes but different extensions come in pairs
        files_df = files_df.sort_values(by=file_name_column)

    if max_num_files is not None and max_num_files > 0:
        files_df = files_df[:max_num_files]

    return files_df


def filter_files_list(filenames: Sequence[Optional[str]],
                      file_extensions: Optional[Collection[str]] = None,
                      column_filter: Optional[AbstractFilterConditionNode] = None,
                      max_num_files: Optional[int] = None,
                      keep_null_values: bool = Config.default_keep_null_values,
                      sort: bool = Config.default_filter_sort,
                      drop_duplicates: bool = Config.default_filter_drop_duplicates,
                      logger: log.Logger = log.DEFAULT_LOGGER) -> List[Optional[str]]:
    """

    :param column_filter:
    :param filenames:
    :param file_extensions:
    :param max_num_files:
    :param keep_null_values:    whether to keep null values in the given :paramref:`filenames`. Cannot be used
                                with :paramref:`drop_duplicates`
    :param sort:
    :param drop_duplicates:     whether to drop duplicate entries in :paramref:`filenames`. Cannot be used with
                                :paramref:`keep_null_values`.
    :param logger:
    :return:
    """
    logger.assert_true(not keep_null_values or not drop_duplicates,
                       "Cannot use keep_null_values and drop_duplicates simultaneously")

    if not keep_null_values:
        filenames = [filename for filename in filenames if filename is not None]
    if len(filenames) == 0:
        return list()

    df = pd.DataFrame(data=filenames, columns=[Config.default_file_name_column])
    if drop_duplicates:
        # TODO this also drops duplicate None values
        # it would require further preprocessing to use it together with keep_null_values
        # possible solution: copy the fileName column to a temporary column, fill all None values there with unique
        # values, drop_duplicates on this temporary column, remove the temporary column
        df = df.drop_duplicates()

    filtered_df = filter_files_df(files_df=df,
                                  file_name_column=Config.default_file_name_column,
                                  file_extensions=file_extensions,
                                  column_filter=column_filter,
                                  max_num_files=max_num_files,
                                  sort=sort,
                                  logger=logger)
    if filtered_df is None:
        return list()

    return filtered_df[Config.default_file_name_column].to_list()
