from typing import Union, Callable

import tensorflow as tf
from tensorflow.python import keras as K


class MaskedLoss(K.losses.LossFunctionWrapper):
    def __init__(
            self,
            loss_function: Callable[[tf.Variable, tf.Variable], tf.Variable],
            masking_value: Union[str, int, float],
            name: str = 'masked_loss',
            reduction: str = tf.keras.losses.Reduction.NONE):
        def _masked_loss(
                y_true: tf.Variable,
                y_pred: tf.Variable,
        ) -> tf.Variable:
            y_true = tf.squeeze(y_true, name="masked_loss__squeezed_y_true")
            y_pred = tf.squeeze(y_pred, name="masked_loss__squeezed_y_pred")

            length_mask = tf.equal(y_true, masking_value,
                                   name="masked_loss__is_masking_value")

            length_mask = tf.cast(length_mask, tf.float32,
                                  name="masked_loss__is_masking_value_float")

            length_mask = tf.math.subtract(
                tf.constant(
                    value=1,
                    dtype=tf.float32
                ), length_mask, name="masked_loss__is_masking_value_inverted")

            lengths = tf.math.reduce_sum(length_mask, axis=-1,
                                         name="masked_loss__sum_to_get_lengths")

            # to also include the first padding character
            lengths = tf.math.add(
                lengths, 1,
                name="masked_loss__sum_to_include_first_padding"
            )

            mask = tf.sequence_mask(
                lengths=lengths,
                maxlen=y_pred.shape[-2],
                # pre-last dimension = padding length
                # last dimension = one-hot-encoded alphabet
                dtype=tf.float32,
                name="masked_loss__create_sequence_mask"
            )

            losses = loss_function(y_true, y_pred)
            losses = tf.math.multiply(
                losses,
                mask,
                name="masked_loss__apply_sequence_mask"
            )

            summed_losses = tf.math.reduce_sum(losses, axis=-1,
                                               name="masked_loss__sum_losses")

            average_losses = tf.math.divide_no_nan(
                summed_losses,
                lengths,
                name="masked_loss__average_losses"
            )

            return average_losses

        super(MaskedLoss, self).__init__(_masked_loss, name=name,
                                         reduction=reduction)
