import gc
import os
from typing import List, Dict, Union, Callable, Any, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from mmproteo.utils import log
from mmproteo.utils.processing import ItemProcessor


class Parquet2DatasetFileProcessor:

    def __init__(self,
                 training_data_columns: List[str],
                 target_data_columns: List[str],
                 padding_lengths: Dict[str, int],
                 padding_characters: Dict[str, Union[str, int, float]],
                 column_normalizations: Dict[str, Callable[[Any], Any]],
                 dataset_dump_path_prefix: str,
                 char_to_idx_mappers: Optional[Dict[str, Dict[str, int]]] = None,
                 char_to_idx_mapping_functions: Optional[Dict[str, Callable[[str], int]]] = None,
                 item_count: int = 0,
                 skip_existing: bool = True,
                 split_on_column_values_of: Optional[List[str]] = None,
                 logger: log.Logger = log.DEFAULT_LOGGER
                 ):
        self.training_data_columns = training_data_columns
        self.target_data_columns = target_data_columns
        self.padding_lengths = padding_lengths
        self.padding_characters = padding_characters
        self.column_normalizations = column_normalizations
        self.char_to_idx_mapping_functions = char_to_idx_mapping_functions
        if self.char_to_idx_mapping_functions is None:
            assert char_to_idx_mappers is not None, \
                "either char_to_idx_mappers or char_to_idx_mapping_functions must be given"
            self.char_to_idx_mapping_functions = {
                column: mapping.get for column, mapping in char_to_idx_mappers.items()
            }
        self.item_count = item_count
        self.dataset_dump_path_prefix = dataset_dump_path_prefix
        self.char_idx_dtype = np.int8
        self.skip_existing = skip_existing
        self.split_on_column_values_of = split_on_column_values_of
        self.logger = logger

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column_normalizations is None:
            return df
        df = df.copy()
        for column, normalize_func in self.column_normalizations.items():
            df[column] = df[column].apply(normalize_func)
        return df

    def pad_array_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df

        df = df.copy()
        for column, padding_length in self.padding_lengths.items():
            item_dtype = df[column].iloc[0].dtype

            df[column] = list(tf.keras.preprocessing.sequence.pad_sequences(
                sequences=df[column],
                maxlen=padding_length,
                padding='post',
                value=self.padding_characters[column],
                dtype=item_dtype
            ))
        return df

    @staticmethod
    def _sequence_to_indices(sequence: Iterable[str],
                             char_to_idx_mapping_func: Callable[[str], int],
                             dtype: type) -> np.ndarray:
        return np.array([char_to_idx_mapping_func(char) for char in sequence],
                        dtype=dtype)

    def sequence_column_to_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.char_to_idx_mapping_functions) == 0:
            return df
        df = df.copy()
        for column, mapping_function in self.char_to_idx_mapping_functions.items():
            df[column] = df[column].apply(lambda seq: self._sequence_to_indices(seq,
                                                                                mapping_function,
                                                                                self.char_idx_dtype))
        return df

    @staticmethod
    def stack_numpy_arrays_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda item: [np.stack(item)])

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        :param df:
        :return: a stacked dataframe (with one single row)
        """
        df = self.normalize_columns(df)
        df = self.pad_array_columns(df)
        df = self.sequence_column_to_indices(df)
        df = self.stack_numpy_arrays_in_dataframe(df)
        return df

    def stacked_df_to_dataset(self, stacked_df: pd.DataFrame) -> tf.data.Dataset:
        assert len(stacked_df) == 1, "all column values should be stacked at this point"
        training_data = tuple(stacked_df[self.training_data_columns].iloc[0])
        target_data = tuple(stacked_df[self.target_data_columns].iloc[0])
        tf_dataset = tf.data.Dataset.from_tensor_slices((training_data, target_data))
        return tf_dataset

    def split_dataframe_by_column_values(self, df: pd.DataFrame, tf_dataset_output_file_path: str) \
            -> List[Tuple[str, pd.DataFrame]]:
        if self.split_on_column_values_of is None:
            return [(tf_dataset_output_file_path, df)]
        if len(self.split_on_column_values_of) == 1:
            value_groups = [((values,), df_split) for values, df_split in df.groupby(self.split_on_column_values_of)]
        else:
            value_groups = [(values, df_split) for values, df_split in df.groupby(self.split_on_column_values_of)]
        results = [(
            os.path.join(
                tf_dataset_output_file_path,
                *[str(value).replace("/", "_") for value in values]),
            df_split.drop(columns=self.split_on_column_values_of)
        ) for values, df_split in value_groups]

        return results

    def convert_df_file_to_dataset_file(self,
                                        df_input_file_path: str,
                                        tf_dataset_output_file_path: str):
        df = pd.read_parquet(df_input_file_path)
        df_splits = self.split_dataframe_by_column_values(df, tf_dataset_output_file_path)
        if len(df_splits) == 0:
            return

        tf_dataset = None

        for path, df_split in df_splits:
            preprocessed_df = self.preprocess_dataframe(df_split)
            tf_dataset = self.stacked_df_to_dataset(preprocessed_df)
            tf.data.experimental.save(dataset=tf_dataset,
                                      path=path,
                                      compression='GZIP')

        self.logger.debug(tf_dataset.element_spec)

    def __call__(self, item: Tuple[int, str]) -> Optional[str]:
        idx, path = item
        tf_dataset_path = os.path.join(self.dataset_dump_path_prefix, path.split(os.path.sep)[-1])

        info_text = f"Processing item {idx + 1}/{self.item_count}: '{path}'"
        if idx % 10 == 0:
            self.logger.info(info_text)
        else:
            self.logger.debug(info_text)

        if self.skip_existing and os.path.exists(tf_dataset_path):
            self.logger.debug(f"Skipped '{path}' because '{tf_dataset_path}' already exists")
            return None

        self.convert_df_file_to_dataset_file(df_input_file_path=path,
                                             tf_dataset_output_file_path=tf_dataset_path)
        gc.collect()

        return tf_dataset_path

    def process(self, parquet_file_paths: Iterable[str], **kwargs) -> List[str]:
        item_processor = ItemProcessor(
            items=enumerate(parquet_file_paths),
            item_processor=self.__call__,
            action_name="parquet2tf_dataset-process",
            subject_name="mzmlid parquet file",
            logger=self.logger,
            **kwargs
        )
        results = list(item_processor.process())
        return results
