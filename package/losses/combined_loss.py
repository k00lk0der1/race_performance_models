import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class CombinedLoss(tf.keras.Loss):
    def __init__(self, a1, a2, eps=0.01, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.a1 = a1
        self.a2 = a2

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

        return (
            tf.keras.losses.cosine_similarity(
                y_true_norm,
                y_true_pred,
                axis=0
            ) + (self.a1 * tf.keras.losses.MSE(y_true, y_pred)
            ) + (self.a2 * tf.keras.losses.MAE(y_true, y_pred)
            ) 
        ) 
    
    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "eps": self.eps,
                "a1": self.a1,
                "a2": self.a2
            }
        )


        return config
