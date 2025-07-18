import tensorflow as tf
from keras.saving import register_keras_serializable
from ..layers import CategoricalEmbeddingLayer

@register_keras_serializable()
class EntityEmbeddingModel(tf.keras.Model):
    def __init__(
        self,
        n_categorical_features,
        categorical_feature_names,
        categorical_feature_ordinalities,
        embedding_dims,
        normalize_embeddings,
        continuous_features_dim,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        self.n_categorical_features = n_categorical_features
        self.categorical_feature_names = categorical_feature_names
        self.categorical_feature_ordinalities = categorical_feature_ordinalities
        self.embedding_dims = embedding_dims
        self.normalize_embeddings = normalize_embeddings
        self.continuous_features_dim = continuous_features_dim

        self.embedding_layers = [
            CategoricalEmbeddingLayer(
                cardinality = self.categorical_feature_ordinalities[ctr],
                embedding_dim = self.embedding_dims[ctr],
                batch_normalization = self.normalize_embeddings[ctr],
                name = self.categorical_feature_names[ctr]
            ) for ctr in range(self.n_categorical_features)
        ]

        self.concatenate_layer = tf.keras.layers.Concatenate(axis=1, name=f"{self.name}_concatenated")

    def call(self, inputs):
        embeddings = [
            self.embedding_layers[ctr](
                inputs[ctr]
            ) for ctr in range(self.n_categorical_features)
        ]
        return self.concatenate_layer(
            embeddings+[inputs[self.n_categorical_features]]
        )
    
    def get_embeddings(self, categorical_feature_name, normalize=True):
        if(categorical_feature_name not in self.categorical_feature_names):
            raise ValueError("Catgeorical feature name not found.")
        
        idx = self.categorical_feature_names.index(categorical_feature_name)
        
        return self.embedding_layers[idx].get_embeddings(normalize=normalize)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_categorical_features": self.n_categorical_features,
                "categorical_feature_names": self.categorical_feature_names,
                "categorical_feature_ordinalities": self.categorical_feature_ordinalities,
                "embedding_dims": self.embedding_dims,
                "normalize_embeddings": self.normalize_embeddings,
                "continuous_features_dim": self.continuous_features_dim
            }
        )
        return config