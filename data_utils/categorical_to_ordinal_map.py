import pandas as pd
import numpy as np

class CategoricalToOrdinalMap:
    def __init__(self):
        self.init = False
    
    def fit(self, series):
        if(self.init):
            raise RuntimeError("Object already has been fit. Cannot call fit/fit_transform again.")
        
        self.unique_sorted_values = series.sort_values().unique()
        self.cardinality = self.unique_sorted_values.shape[0]
        self.unique_set = set(self.unique_sorted_values)

        self.forward_map = dict(
            [
                (
                    self.unique_sorted_values[x],
                    x
                ) for x in range(self.cardinality)
            ]
        )

        self.backward_map = dict(
            [
                (
                    x,
                    self.unique_sorted_values[x]
                ) for x in range(self.cardinality)
            ]
        )

        self.init = True
    
    def transform(self, series):

        if(not self.init):
            raise RuntimeError("Object has been not been fit. Call fit/fit_transform before calling transform.")
        
        return series.map(self.forward_map)
        
    
    def inverse_transform(self, series):

        if(not self.init):
            raise RuntimeError("Object has been not been fit. Call fit/fit_transform before calling inverse_transform.")

        return series.map(self.backward_map)
    
    def check_membership(self, value):
        return value in self.unique_set
    
    def fit_transform(self, series):

        self.fit(series)
        
        return self.transform(series)
    
