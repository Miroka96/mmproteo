import tensorflow as tf

from typing import Union, Tuple, Any, List


class InputLayerConfiguration:
    def __init__(self, name: str, shape: Union[int, Tuple[Any]], mask_value: Any):
        self.name = name
        self.shape = shape
        self.mask_value = mask_value

    def get_shape_tuple(self) -> Tuple[Any]:
        if not isinstance(self.shape, tuple):
            return self.shape,
        return self.shape


def create_masked_input_layers(input_layer_configurations: List[InputLayerConfiguration]) \
        -> Tuple[List[tf.keras.layers.Input], List[tf.keras.layers.Masking]]:
    named_input_layers = {
        config.name: tf.keras.layers.Input(
            shape=config.get_shape_tuple(),
            name=config.name
        ) for config in input_layer_configurations
    }
    masked_input_layers = {
        config.name: tf.keras.layers.Masking(
            mask_value=config.mask_value,
            name=f"masked_{config.name}",
        )(named_input_layers[config.name])
        for config in input_layer_configurations
    }
    named_input_layers_list = [named_input_layers[config.name] for config in input_layer_configurations]
    masked_input_layers_list = [masked_input_layers[config.name] for config in input_layer_configurations]
    return (named_input_layers_list, masked_input_layers_list)
