import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class NormalizedCosineSimilarity(tf.keras.Loss):
    def __init__(self, eps=0.01, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, y_true, y_pred):
        y_true_norm = (
            y_true - 
            tf.math.reduce_mean(y_true, keepdims=True)
        )/tf.math.sqrt(
            tf.math.reduce_variance(y_true, keepdims=True) + self.eps
        )

        y_true_pred = (
            y_pred - 
            tf.math.reduce_mean(y_pred, keepdims=True)
        )/tf.math.sqrt(
            tf.math.reduce_variance(y_pred, keepdims=True) + self.eps
        )

        return tf.keras.losses.cosine_similarity(
            y_true_norm,
            y_true_pred,
            axis=0
        )
    
    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "eps": self.eps,
            }
        )


        return config
