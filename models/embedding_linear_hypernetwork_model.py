
import tensorflow as tf
from keras.saving import register_keras_serializable
from .linear_hypernetwork_model import LinearHyperNetworkModel
from .entity_embedding_model import EntityEmbeddingModel

@register_keras_serializable()
class EmbeddingLinearHypernetworkModel(tf.keras.Model):
    def __init__(
        self,
        context_embedding_model_kwargs,
        covariate_embedding_model_kwargs,
        linear_hypernetwork_model_kwargs,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        self.context_embedding_model_kwargs = context_embedding_model_kwargs
        self.covariate_embedding_model_kwargs = covariate_embedding_model_kwargs
        self.linear_hypernetwork_model_kwargs = linear_hypernetwork_model_kwargs

        self.context_embedding_model = EntityEmbeddingModel(
            **context_embedding_model_kwargs
        )

        self.covariate_embedding_model = EntityEmbeddingModel(
            **covariate_embedding_model_kwargs
        )

        self.linear_hypernetwork_model = LinearHyperNetworkModel(
            **linear_hypernetwork_model_kwargs
        )

        self.input_split_index = self.context_embedding_model_kwargs["n_categorical_features"]+1
        #print(self.input_split_index)

    
    def call(self, inputs):
        context_embedding_inputs = inputs[:self.input_split_index]
        covariate_embedding_inputs = inputs[self.input_split_index:]

        context_embedding = self.context_embedding_model(context_embedding_inputs)
        covariate_embedding = self.covariate_embedding_model(covariate_embedding_inputs)

        return self.linear_hypernetwork_model(
            [
                context_embedding,
                covariate_embedding
            ]
        )
        
    
    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "context_embedding_model_kwargs": self.context_embedding_model_kwargs,
                "covariate_embedding_model_kwargs": self.covariate_embedding_model_kwargs,
                "linear_hypernetwork_model_kwargs": self.linear_hypernetwork_model_kwargs,
                #"input_split_index":self.input_split_index
            }
        )


        return config