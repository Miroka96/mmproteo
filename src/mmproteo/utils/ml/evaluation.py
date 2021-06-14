from typing import Callable, Optional, List, Union, Tuple, Iterable, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from mmproteo.utils.utils import unzip


class SequenceEvaluator:
    def __init__(
            self,
            dataset: tf.data.Dataset,
            decode_func: Optional[Callable[[np.ndarray], np.ndarray]],
            batch_size: int = 32,
            separator: str = " ",
            padding_character: str = "_",
            prediction_col_name: str = "predicted",
            true_value_col_name: str = "true",
    ):
        self.dataset = dataset
        self.decode_func = decode_func
        self.batch_size = batch_size
        self.separator = separator
        self.padding_character = padding_character
        self.prediction_col_name = prediction_col_name
        self.true_value_col_name = true_value_col_name

    def evaluate(
            self,
            model: tf.keras.Model,
            steps: Optional[int] = 0,
    ) -> Union[float, List[float]]:
        if steps is None:
            return model.evaluate(self.dataset)

        if steps == 0:
            steps = 40000 / self.batch_size

        return model.evaluate(self.dataset.repeat(), steps=steps)

    def shorten_sequences_to_lengths_of_other_sequences_with_separator(
            self,
            sequences: pd.Series,
            other_sequences: pd.Series,
            length_offset: int = 0,
            split_by_separator: bool = True,
    ) -> pd.Series:
        other_sequences = other_sequences.str.rstrip(self.padding_character)

        if split_by_separator:
            other_sequences = other_sequences.str.rstrip(self.separator)
            other_sequences = other_sequences.str.split(self.separator)

            def shorten_sequence(sequence, length):
                return self.separator.join(
                    sequence.split(self.separator)[:length]
                )
        else:
            def shorten_sequence(sequence, length):
                return sequence[:length]

        lengths = other_sequences.str.len()
        lengths += length_offset

        return sequences.combine(
            other=lengths,
            func=shorten_sequence,
        )

    def evaluate_visually(
            self,
            model: tf.keras.Model,
            sample_size: int = 20,
            keep_separator: bool = True,
            split_by_separator: bool = True,
    ) -> Tuple[pd.DataFrame, Tuple[Iterable, Iterable, Any]]:
        eval_ds = self.dataset.unbatch().batch(1).take(sample_size)
        x_eval, y_eval = unzip(eval_ds.as_numpy_iterator())
        y_pred = model.predict(eval_ds)

        eval_df = pd.DataFrame(
            data=zip(
                self.decode(y_pred, onehot=True),
                self.decode(y_eval, onehot=False)
            ),
            columns=[self.prediction_col_name, self.true_value_col_name]
        )

        eval_df[self.prediction_col_name] = \
            self.shorten_sequences_to_lengths_of_other_sequences_with_separator(
                sequences=eval_df[self.prediction_col_name],
                other_sequences=eval_df[self.true_value_col_name],
                length_offset=1,
                split_by_separator=split_by_separator,
            )

        if not keep_separator:
            eval_df = eval_df.applymap(lambda s: s.replace(self.separator, ""))

        eval_df[self.true_value_col_name] = eval_df[self.true_value_col_name]\
            .str.rstrip(
                self.padding_character + self.separator
            )

        return (eval_df, (x_eval, y_eval, y_pred))

    @staticmethod
    def decode_onehot(array: np.ndarray) -> np.ndarray:
        return np.argmax(array, axis=-1)

    def concat_letter_rows(self, array: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda row: self.separator.join(row),
                                   axis=-1, arr=array)

    def decode(
            self,
            array: np.ndarray,
            onehot: bool = True,
    ) -> np.ndarray:
        if onehot:
            array = self.decode_onehot(array)
        if self.decode_func is not None:
            array = self.decode_func(array)
        array = self.concat_letter_rows(array)
        if not onehot:
            array = np.apply_along_axis(lambda row: row[0], axis=-1, arr=array)
        return array