import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
import wandb

from package.data_utils.minibatch import PickledMinibatch
from package.data_utils import build_model_Xy_v2
from package.models import EmbeddingLinearHypernetworkModel, EmbeddingSequentialModel
from package.losses import CombinedLoss, NormalizedCosineSimilarity

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import argparse
parser = argparse.ArgumentParser(description='Run Experiments on Dataset')

parser.add_argument(
    '--target_variable',
    type=str,
    choices=["SPEED", "FINISH_TIME"],
    default='SPEED',
    help='The target variable to use.'
)

parser.add_argument(
    '--epochs',
    type=int,
    required=True,
    help='Number of epochs to train the model.'
)

parser.add_argument(
    '--model_type',
    type=str,
    choices=["SEQUENTIAL", "HYPERNETWORK"],
    default='HYPERNETWORK',
    help='Type of model to use.'
)

parser.add_argument(
    '--loss_function',
    type=str,
    choices=["MSE", "MAE", "COSINE_SIMILARITY", "COMBINED_LOSS"],
    default='MAE',
    help='Type of model to use.'
)

parser.add_argument(
    '--data_folder_path',
    type=str,
    required=True,
    help='Path for data folder.'
)

parser.add_argument(
    '--secrets_file',
    type=str,
    required=True,
    help='Path for secret files with wandb api.'
)

parser.add_argument(
    '--verbose',
    type=int,
    default=2,
    help='Verbosity of Keras fit call.'
)


args = parser.parse_args()

wandb_api_key = json.loads(
    open(
        args.secrets_file
    ).read()
)["wandb_api_key"]

wandb_project_name = json.loads(
    open(
        args.secrets_file
    ).read()
)["wandb_project_name"]

wandb.login(key=wandb_api_key)

model_configurations = json.load(
    open(
        os.path.join(
            args.data_folder_path,
            "model_config.json"
        )
    )
)


wandb.init(
    project=wandb_project_name,
    config={
        "target_variable": args.target_variable,
        "epochs": args.epochs,
        "loss_function": args.loss_function,
        "model_type": args.model_type,
        "race_embedding_model_kwargs": model_configurations['race_embedding_model_kwargs'],
        "horse_embedding_model_kwargs":model_configurations['horse_embedding_model_kwargs'],
        "linear_hyper_model_kwargs":model_configurations['linear_hyper_model_kwargs'],
        "sequential_model_kwargs":model_configurations['sequential_model_kwargs']
    }
)


train = pd.read_parquet(
    os.path.join(
        args.data_folder_path,
        f"TRAIN_{args.target_variable}.pq"
    )
)

test = pd.read_parquet(
    os.path.join(
        args.data_folder_path,
        f"TEST_{args.target_variable}.pq"
    )
)

validation = pd.read_parquet(
    os.path.join(
        args.data_folder_path,
        f"VALIDATION_{args.target_variable}.pq"
    )
)

train_minibatch_obj = PickledMinibatch()
validation_minibatch_obj = PickledMinibatch()

train_minibatch_obj.read_from_pickle(
    os.path.join(
        args.data_folder_path,
        f"TRAIN_BATCHES_{args.target_variable}.pkl"
    )
)

validation_minibatch_obj.read_from_pickle(
    os.path.join(
        args.data_folder_path,
        f"VALIDATION_BATCHES_{args.target_variable}.pkl"
    )
)

loss_functions = {
    "MSE": tf.keras.losses.MeanSquaredError(),
    "MAE": tf.keras.losses.MeanAbsoluteError(),
    "COSINE_SIMILARITY": NormalizedCosineSimilarity(),
    "COMBINED_LOSS": CombinedLoss(a1=0,a2=1)
}

loss_function = loss_functions[args.loss_function]


if(args.model_type=="HYPERNETWORK"):
    model = EmbeddingLinearHypernetworkModel(
        model_configurations['race_embedding_model_kwargs'],
        model_configurations['horse_embedding_model_kwargs'],
        model_configurations['linear_hyper_model_kwargs'],
    )
elif(args.model_type=="SEQUENTIAL"):
    model = EmbeddingSequentialModel(
        model_configurations['race_embedding_model_kwargs'],
        model_configurations['horse_embedding_model_kwargs'],
        model_configurations['sequential_model_kwargs'],        
    )

callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ModelCheckpoint(
        "best_validation_loss.keras",
        save_weights_only=False,
        monitor='val_loss',
        mode="min",
        save_best_only=True
    ),
    WandbModelCheckpoint(
        "best_validation_loss.keras",
        save_weights_only=False,
        monitor='val_loss',
        mode="min",
        save_best_only=True
    ),
    WandbMetricsLogger()
]

model.compile(
    optimizer="adagrad",
    loss = loss_function
)

model_fit_history = model.fit(
    x=train_minibatch_obj.generator(),
    epochs = args.epochs, steps_per_epoch = train_minibatch_obj.n_races,
    validation_data=validation_minibatch_obj.generator(),
    validation_steps=validation_minibatch_obj.n_races,
    callbacks=callbacks,
    verbose=args.verbose,
)
