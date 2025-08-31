import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SequentialModel(tf.keras.Model):
    def __init__(
        self,
        layer_widths,
        activations,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        self.layer_widths = layer_widths
        self.activations = activations

        self.dense_layers = (
            [
                tf.keras.layers.Dense(
                    units = layer_width,
                    activation = activation
                ) for layer_width, activation in zip(self.layer_widths, self.activations)
            ] + [
                tf.keras.layers.Dense(
                    units = 1,
                    activation = "linear"
                )
            ]
        )
    
    def call(self, inputs):
        x = inputs
        
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_widths": self.layer_widths,
                "activations": self.activations
            }
        )
        return config