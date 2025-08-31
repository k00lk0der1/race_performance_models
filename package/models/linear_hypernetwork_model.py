
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class LinearHyperNetworkModel(tf.keras.Model):
    def __init__(
        self,
        layer_widths,
        activations,
        linear_model_dim,
        L1L2_regularization_penalty_kwargs = {
            "l1":0,
            "l2":0
        },
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        self.layer_widths = layer_widths
        self.activations = activations
        self.linear_model_dim = linear_model_dim
        self.L1L2_regularization_penalty_kwargs = L1L2_regularization_penalty_kwargs

        self.dense_layers = (
            [
                tf.keras.layers.Dense(
                    units = layer_width,
                    activation = activation
                ) for layer_width, activation in zip(self.layer_widths, self.activations)
            ] + [
                tf.keras.layers.Dense(
                    units = self.linear_model_dim,
                    activation = "linear",
                    activity_regularizer = tf.keras.regularizers.L1L2(
                        **self.L1L2_regularization_penalty_kwargs
                    )
                )
            ]
        )
    
    def call(self, inputs):
        context = inputs[0]
        covariates = inputs[1]

        beta = context
        for dense_layer in self.dense_layers:
            beta = dense_layer(beta)
        
        output = tf.keras.ops.sum(
            tf.keras.ops.multiply(
                beta,
                covariates
            ),
            keepdims=True,
            axis=1
        )

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_widths": self.layer_widths,
                "activations": self.activations,
                "linear_model_dim": self.linear_model_dim,
                "L1L2_regularization_penalty_kwargs":self.L1L2_regularization_penalty_kwargs,
            }
        )
        return config