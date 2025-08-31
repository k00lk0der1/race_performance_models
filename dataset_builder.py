import sqlite3
import pandas as pd
import numpy as np
import datetime, dateutil
import json

from package.data_utils import CategoricalToOrdinalMap
from package.data_utils import get_sql_query, build_model_Xy_v2, get_train_test_validation_split
from package.data_utils.minibatch import PickledMinibatch

import argparse
parser = argparse.ArgumentParser(description='Dataset and MiniBatch builder')

parser.add_argument(
    '--target_variable',
    type=str,
    choices=["SPEED", "FINISH_TIME"],
    default='SPEED',
    help='The target variable to use.'
)
parser.add_argument(
    '--sql_file',
    type=str,
    required=True,
    help='Path to the SQLite file.'
)

args = parser.parse_args()

sql_connection = sqlite3.connect(args.sql_file)

sql_params = {
    "min_winner_time":30,
    "min_winner_speed":10,
    "max_winner_speed":25,
    "min_runner_age":0,
    "max_runner_age":50,
    "min_total_prize":10
}


races = pd.read_sql_query(
    get_sql_query("historic_races_query_v2.sql"),
    sql_connection,
    params = sql_params
)

races['date'] = races['date'].apply(lambda x : dateutil.parser.parse(x))


runners = pd.read_sql_query(
    get_sql_query("historic_runners_query_v2.sql"),
    sql_connection,
    params = sql_params
)

runners.loc[(runners['distance_behind_winner'].isna() & runners['finish_position']==1), 'distance_behind_winner'] = 0
runners = runners.drop(['distance_travelled', 'finish_position'], axis=1).dropna()

merged_dataset = pd.merge(runners, races, "inner", on="race_id")

runner_categorical_variables = [
    'runner_id',
    'runner_gender',
    'runner_breeding_country'
]

race_categorical_variables = [
    'race_course',
    'race_type',
    'race_going_condition',
    'race_direction'
]

runner_cat_to_ord_maps = dict(
    [
        (
            cat_var_name,
            CategoricalToOrdinalMap()
            
        ) for cat_var_name in runner_categorical_variables
    ]
)

races_cat_to_ord_maps = dict(
    [
        (
            cat_var_name,
            CategoricalToOrdinalMap()
            
        ) for cat_var_name in race_categorical_variables
    ]
)

for cat_var_name in runner_categorical_variables:
    merged_dataset[cat_var_name] = runner_cat_to_ord_maps[cat_var_name].fit_transform(
        merged_dataset[cat_var_name]
    )

for cat_var_name in race_categorical_variables:
    merged_dataset[cat_var_name] = races_cat_to_ord_maps[cat_var_name].fit_transform(
        merged_dataset[cat_var_name]
    )

if(args.target_variable=="SPEED"):
    merged_dataset['y'] = np.log(
        (
            (
                merged_dataset['distance_yards'] - 
                (merged_dataset['distance_behind_winner']/3)
            ) / (
                merged_dataset['winning_time_secs']
            ) 
        )
    )
elif(args.target_variable=="FINISH_TIME"):
    merged_dataset['y'] = np.log(merged_dataset['distance_yards']/
        (
            (
                merged_dataset['distance_yards'] - 
                (merged_dataset['distance_behind_winner']/3)
            ) / (
                merged_dataset['winning_time_secs']
            ) 
        )
    )
else:
    raise RuntimeError("Invalid Reponse Variable")


(
    train,
    validation,
    test
) = get_train_test_validation_split(merged_dataset, 0.8)


relevant_columns = [
    'runner_id',
    'runner_gender',
    'race_course',
    'race_type',
    'race_going_condition',
    'race_direction'
]

assert((merged_dataset[relevant_columns].nunique()==train[relevant_columns].nunique()).astype(int).prod()==1)

train.to_parquet(f"TRAIN_{args.target_variable}.pq")
test.to_parquet(f"TEST_{args.target_variable}.pq")
validation.to_parquet(f"VALIDATION_{args.target_variable}.pq")

train_minibatch_obj = PickledMinibatch()
validation_minibatch_obj = PickledMinibatch()

train_minibatch_obj.generate_batches(train, build_model_Xy_v2, runner_categorical_variables, race_categorical_variables)
validation_minibatch_obj.generate_batches(validation, build_model_Xy_v2, runner_categorical_variables, race_categorical_variables)

train_minibatch_obj.write_to_pickle(f"TRAIN_BATCHES_{args.target_variable}.pkl")
validation_minibatch_obj.write_to_pickle(f"VALIDATION_BATCHES_{args.target_variable}.pkl")

gean_data = pd.read_sql_query(
    get_sql_query("gean_similairty_query_v1.sql"),
    sql_connection
)

gean_data = gean_data.dropna()

gean_data['dam_id'] = gean_data['dam_id'].apply(int)
gean_data['sire_id'] = gean_data['sire_id'].apply(int)

runners_with_encoding = runner_cat_to_ord_maps['runner_id'].unique_set
gean_data = gean_data[(
    gean_data['runner_id'].apply(lambda x : x in runners_with_encoding) &
    gean_data['dam_id'].apply(lambda x : x in runners_with_encoding) &
    gean_data['sire_id'].apply(lambda x : x in runners_with_encoding) 
)]

gean_data["runner_id"] = runner_cat_to_ord_maps['runner_id'].transform(gean_data["runner_id"])
gean_data["dam_id"] = runner_cat_to_ord_maps['runner_id'].transform(gean_data["dam_id"])
gean_data["sire_id"] = runner_cat_to_ord_maps['runner_id'].transform(gean_data["sire_id"])

gean_data.to_parquet("GEAN_DATA.pq")

horse_embedding_model_kwargs = {
    "n_categorical_features":len(runner_categorical_variables),
    "categorical_feature_names":runner_categorical_variables,
    "categorical_feature_ordinalities":[train[cat_var_name].nunique() for cat_var_name in runner_categorical_variables],
    "embedding_dims": [4] + [4 for _ in runner_categorical_variables[1:]],
    "normalize_embeddings": [False for _ in runner_categorical_variables],
    "continuous_features_dim":4,
    "name":"horse"
}

race_embedding_model_kwargs = {
    "n_categorical_features":len(race_categorical_variables),
    "categorical_feature_names":race_categorical_variables,
    "categorical_feature_ordinalities":[train[cat_var_name].nunique() for cat_var_name in race_categorical_variables],
    "embedding_dims":[4 for _ in race_categorical_variables],
    "normalize_embeddings": [False for _ in race_categorical_variables],
    "continuous_features_dim":2,
    "name":"race"
}

linear_hyper_model_kwargs = {
    "layer_widths":[128],
    "activations":["tanh"],
    "linear_model_dim":sum(horse_embedding_model_kwargs["embedding_dims"])+horse_embedding_model_kwargs["continuous_features_dim"],
    "L1L2_regularization_penalty_kwargs":{"l1":0, "l2":0}
}

sequential_model_kwargs = {
    "layer_widths":[128],
    "activations":["tanh"]
}

model_configurations = {
    "horse_embedding_model_kwargs" : horse_embedding_model_kwargs,
    "race_embedding_model_kwargs" : race_embedding_model_kwargs,
    "linear_hyper_model_kwargs" : linear_hyper_model_kwargs,
    "sequential_model_kwargs" : sequential_model_kwargs,
    
}

model_config_file = open("model_config.json", "w")
model_config_file.write(json.dumps(model_configurations))