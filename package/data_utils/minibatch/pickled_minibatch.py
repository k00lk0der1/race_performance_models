import numpy as np
import pickle

#This file is in serious need for error handling.
#Would generate runtime/logical errors if used improperly.

def get_races_with_mt_2_runners(dataset):
    runner_count = dataset.groupby("race_id")['runner_id'].count()
    races_with_mt_2_runners = set(runner_count[runner_count>2].index)
    return races_with_mt_2_runners

class PickledMinibatch:
    def __init__(self):
        self.init=False
    
    def generate_batches(self, dataset, dataset_builder_function, runner_categorical_variables, race_categorical_variables):
        races_with_mt_2_runners = get_races_with_mt_2_runners(dataset)
        data_subset = dataset[dataset['race_id'].apply(lambda x : x in races_with_mt_2_runners)]
        race_subsets = data_subset.groupby("race_id")

        self.batches = []
        
        for race_subset in race_subsets:
            subset = race_subset[1]

            batch = dataset_builder_function(
                subset,
                runner_categorical_variables,
                race_categorical_variables
            )

            self.batches.append((tuple(batch[0]), batch[1]))
        
        self.n_races = len(self.batches)
        self.init = True

    
    def write_to_pickle(self, pickle_file):
        fo = open(pickle_file, "wb")
        pickle.dump(self.batches, fo)
        fo.close()

    def read_from_pickle(self, pickle_file):
        fo = open(pickle_file, "rb")
        self.batches = pickle.load(fo)
        fo.close()
        
        self.n_races = len(self.batches)
        self.init = True
    
    def generator(self, save_to_pickle=False):

        if(not self.init):
            raise RuntimeError("Object not initialized. Either call generate_batches or read_from_pickle.")
        
        while True:
            for i in np.random.choice(range(self.n_races), size=self.n_races, replace=False):
                yield self.batches[i]




