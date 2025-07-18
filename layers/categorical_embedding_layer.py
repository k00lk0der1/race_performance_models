import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class CategoricalEmbeddingLayer(tf.keras.Layer):
    def __init__(
        self,
        cardinality,
        embedding_dim,
        batch_normalization=False,
        epsilon=0.01,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        self.cardinality = cardinality
        self.embedding_dim = embedding_dim
        self.batch_normalization = batch_normalization
        self.epsilon = epsilon

        self.ordinal_to_one_hot = tf.keras.layers.CategoryEncoding(
            num_tokens = self.cardinality,
            output_mode="one_hot"
        )

        self.embedding_dense = tf.keras.layers.Dense(
            units = self.embedding_dim,
            activation = "linear",
            use_bias = False,
            #kernel_initializer=tf.keras.initializers.RandomNormal(
            #    mean=0,
            #    stddev=1
            #)
        )

    
    def call(self, inputs):
        one_hot_encoding = self.ordinal_to_one_hot(inputs)
        
        category_embedding = self.embedding_dense(one_hot_encoding)
        
        if(self.batch_normalization):
            category_embedding = (
                (
                    category_embedding - 
                    tf.math.reduce_mean(category_embedding, axis=0, keepdims=True)
                ) / tf.sqrt(
                    tf.math.reduce_variance(category_embedding, axis=0, keepdims=True) + 
                    self.epsilon
                )
            )
        
        return category_embedding
    
    def get_embeddings(self, normalize=True):
        embeddings = self.embedding_dense.weights[0].value.numpy()
        if(normalize):
            embeddings = (
                (
                    embeddings - 
                    embeddings.mean(axis=0, keepdims=True)
                ) / 
                embeddings.std(axis=0, keepdims=True)
            )
        return embeddings
    
    def get_config(self):
        config = super().get_config()
        
        config.update({
            "cardinality": self.cardinality,
            "embedding_dim": self.embedding_dim,
            "batch_normalization": self.batch_normalization,
            "epsilon":self.epsilon
        })
        
        return config
        
