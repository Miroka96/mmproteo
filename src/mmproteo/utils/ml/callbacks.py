import datetime
import os
import shutil
from typing import Union, List

import tensorflow as tf
from tqdm.keras import TqdmCallback


def create_tensorboard_callback(
        tensorboard_log_dir: str = "logs",
        training_type: str = "fit",
        keep_logs: bool = True,
) -> tf.keras.callbacks.TensorBoard:
    if not keep_logs:
        try:
            shutil.rmtree(tensorboard_log_dir)
        except FileNotFoundError:
            pass
    return tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(
            tensorboard_log_dir,
            training_type,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
        histogram_freq=1,
    )


# Parts of the following code are based on code by Sven Giese:
# https://github.com/Rappsilber-Laboratory/xiRT/blob/master/xirt/xirtnet.py
# (28.06.21; method 'xiRTNET.get_callbacks')
# https://doi.org/10.5281/zenodo.4669957

def create_progressbar_callback(verbose: int = 0) -> TqdmCallback:
    return TqdmCallback(verbose=verbose)


def create_reduce_learning_rate_callback(
        monitor: str = 'val_loss',
        reduce_lr_factor: float = 0.1,
        reduce_lr_patience: int = 10,
        verbose: int = 1,
        min_delta: float = 1e-4,
        mode: str = 'min',
) -> tf.keras.callbacks.ReduceLROnPlateau:
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        verbose=verbose,
        min_delta=min_delta,
        mode=mode
    )


def create_early_stopping_callback(
        monitor: str = 'val_loss',
        mode: str = 'min',
        verbose: int = 1,
        patience: int = 10,
        restore_best_weights: bool = True,
) -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        verbose=verbose,
        patience=patience,
        restore_best_weights=restore_best_weights
    )


def create_checkpoint_callback(
        filepath: str,
) -> tf.keras.callbacks.ModelCheckpoint:
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
    )


def create_log_csv_callback(
        filename: str
) -> tf.keras.callbacks.CSVLogger:
    return tf.keras.callbacks.CSVLogger(
        filename=filename,
    )


def create_callbacks(
        tensorboard: Union[bool, tf.keras.callbacks.TensorBoard] = False,
        progressbar: Union[bool, TqdmCallback] = False,
        reduce_lr: Union[bool, tf.keras.callbacks.ReduceLROnPlateau] = False,
        early_stopping: Union[bool, tf.keras.callbacks.EarlyStopping] = False,
        checkpoints: Union[bool, tf.keras.callbacks.ModelCheckpoint] = False,
        csv: Union[bool, tf.keras.callbacks.CSVLogger] = False,
        base_path: str = ".",
) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = []

    if tensorboard is True:
        tensorboard = create_tensorboard_callback(
            os.path.join(base_path, 'tensorboard')
        )
    if tensorboard:
        callbacks.append(tensorboard)

    if progressbar is True:
        progressbar = create_progressbar_callback()
    if progressbar:
        callbacks.append(progressbar)

    if reduce_lr is True:
        reduce_lr = create_reduce_learning_rate_callback()
    if reduce_lr:
        callbacks.append(reduce_lr)

    if early_stopping is True:
        early_stopping = create_early_stopping_callback()
    if early_stopping:
        callbacks.append(early_stopping)

    if checkpoints is True:
        checkpoints = create_checkpoint_callback(
            filepath=os.path.join(base_path, 'checkpoints')
        )
    if checkpoints:
        callbacks.append(checkpoints)

    if csv is True:
        csv = create_log_csv_callback(
            filename=os.path.join(base_path, 'log.csv')
        )
    if csv:
        callbacks.append(csv)

    return callbacks
