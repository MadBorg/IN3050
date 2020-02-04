# for debug
import IPython as IP
# --
import pandas as pd
import numpy as np
import itertools
import multiprocessing

import data as city_data      

class exhaustiveSearch:

    def __init__(self, data):
        self.data = data
        self.columns = data.columns[:]
        self.cities = [i for i in range(len(self.columns))]

    def __call__(self):
        cities = self.cities
        start_path = cities + [cities[0]]
        s = 0
        for i in range(len(start_path)-1):
            s += self.data.iloc[start_path[i], start_path[i+1]]
        self.bestPath = start_path
        self.bestScore = s


        print(f"bestPath: {self.bestPath}, bestScore: {self.bestScore}")
        current = 0

        with multiprocessing.Pool(maxtasksperchild=500) as pool: # default is optimal number of processes
            pool.map(self._fit, itertools.permutations(cities, len(cities)))

        return self.bestPath, self.bestScore

    def _fit(self, perm):
        path = list(perm) + [perm[0]]
        s = 0
        for i in range(len(path)-1):
            s += self.data.iloc[path[i], path[i+1]]
        
        self.bestScore = s
        self.bestPath = path
        
    
    


if __name__ == "__main__":
    tmp = exhaustiveSearch(city_data.data_subset(city_data.path_to_datafile, 5))
    print(tmp())
