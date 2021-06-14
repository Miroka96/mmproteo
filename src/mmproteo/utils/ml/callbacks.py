import datetime
import os
import shutil

import tensorflow as tf

def create_tensorboard_callback(
        tensorboard_log_dir: str = "logs",
        training_type: str = "fit",
        keep_logs: bool = True,
):
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