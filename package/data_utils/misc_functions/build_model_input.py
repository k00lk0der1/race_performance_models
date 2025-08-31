import numpy as np

def build_model_Xy_v1(dataframe, runner_categorical_variables, race_categorical_variables):
    return (
        [
            dataframe[[cat_var_name]].values for cat_var_name in race_categorical_variables
        ] + [
                np.concatenate(
                    [
                        dataframe[['distance_yards']].values,
                        np.log(dataframe[['distance_yards']].values)
                    ],
                    axis=-1
                )
            
        ] + [
            dataframe[[cat_var_name]].values for cat_var_name in runner_categorical_variables
        ] + [
                np.concatenate(
                    [
                        dataframe[['runner_age']].values,
                        np.power(dataframe[['runner_age']].values, 2),
                        np.power(dataframe[['runner_age']].values, 3),
                        np.ones(shape=dataframe[['runner_id']].values.shape)
                    ],
                    axis=-1
                )
        ],
        dataframe[['y']].values
    ) 

def build_model_Xy_v2(dataframe, runner_categorical_variables, race_categorical_variables):
    return (
        [
            dataframe[[cat_var_name]].values for cat_var_name in race_categorical_variables
        ] + [
                np.concatenate(
                    [
                        dataframe[['distance_yards']].values,
                        np.log(dataframe[['distance_yards']].values),
                        np.log(dataframe[['added_money']].values)
                    ],
                    axis=-1
                )
            
        ] + [
            dataframe[[cat_var_name]].values for cat_var_name in runner_categorical_variables
        ] + [
                np.concatenate(
                    [
                        dataframe[['runner_age']].values,
                        np.power(dataframe[['runner_age']].values, 2),
                        np.power(dataframe[['runner_age']].values, 3),
                        np.ones(shape=dataframe[['runner_id']].values.shape)
                    ],
                    axis=-1
                )
        ],
        dataframe[['y']].values
    ) 
