import numpy as np
import pandas as pd

def get_train_test_validation_split(merged_dataset, train_percentage):
    """
    merged_dataset: pd.DataFrame with columns necessarily including
    race_id, date (datetime.datetime type) and runner_id
    train_percentage: float in range [0,1].
    Return train, test, validation with train
    length approximately equal to len(merged_dataset*train_percentage)
    and test and validation lengths roughly equal to len(merged_dataset*(1-train_percentage))/2

    The goal is to do the race wise split of the dataset while also ensuring that all runners
    that appear in the test and validation also appear atleast once in the train set
    """

    merged_dataset = merged_dataset.sort_values("date").reset_index(drop=True)

    runners_n_prior_runs = dict([])
    runners_n_prior_runs_datewise = []

    for runner_race_combo in merged_dataset.iterrows():
        if(runner_race_combo[1]['runner_id'] not in runners_n_prior_runs.keys()):
            runners_n_prior_runs[runner_race_combo[1]['runner_id']] = 0
        runners_n_prior_runs_datewise.append(
            runners_n_prior_runs[runner_race_combo[1]['runner_id']]
        )
        runners_n_prior_runs[runner_race_combo[1]['runner_id']] = (
            1 + 
            runners_n_prior_runs[runner_race_combo[1]['runner_id']]
        )

    merged_dataset['runners_n_prior_runs_dw'] = runners_n_prior_runs_datewise
    merged_dataset.groupby('race_id')['runners_n_prior_runs_dw'].min()

    n_prior_runs_min_count_df = merged_dataset.groupby(
        "race_id"
    ).agg(
        {
            "runners_n_prior_runs_dw" : [
                "min",
                "count"
            ]
    })

    n_prior_runs_min_count_df = n_prior_runs_min_count_df.droplevel(level=0, axis=1)
    n_prior_runs_min_count_df.sort_values("min", inplace=True)
    n_prior_runs_min_count_df.reset_index(drop=True)
    n_prior_runs_min_count_df["count_cumsum"]  = n_prior_runs_min_count_df["count"].cumsum()

    min_train_set_length = int(merged_dataset.shape[0]*train_percentage)
    min_val_set_length = int(merged_dataset.shape[0]*(1-train_percentage))//2


    train_end_index = (
        n_prior_runs_min_count_df["count_cumsum"] > min_train_set_length
    ).argmax()

    assert(
        n_prior_runs_min_count_df["min"].iloc[train_end_index]>0,
        "Increase percentage of samples in train dataset"
    )

    val_end_index = (
        n_prior_runs_min_count_df["count_cumsum"] > (
            min_train_set_length+min_val_set_length
        )
    ).argmax()

    train_races = set(n_prior_runs_min_count_df.iloc[:train_end_index].index)
    val_races = set(n_prior_runs_min_count_df.iloc[train_end_index:val_end_index].index)
    test_races = set(n_prior_runs_min_count_df.iloc[val_end_index:].index)

    train_dataset = merged_dataset[merged_dataset['race_id'].apply(lambda x : x in train_races)]
    val_dataset = merged_dataset[merged_dataset['race_id'].apply(lambda x : x in val_races)]
    test_dataset = merged_dataset[merged_dataset['race_id'].apply(lambda x : x in test_races)]
        
    return (
        train_dataset,
        val_dataset,
        test_dataset
    )
