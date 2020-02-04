# for debug
import IPython as IP
# --
import pandas as pd
import numpy as np
import itertools
import multiprocessing

import data as city_data                                    



def fit(path, data):
    cum = 0;
    for i in range(len(path)-1):
        cum += data.iloc[path[i], path[i+1]]
    return cum
    
def fun(perm, data):
    path = list(perm) + [perm[0]]
    current = fit(path, data)
    return current, path

def exhaustiveSearch(data):
    cities = data.columns
    n = len(cities)
    cities_int = [i for i in range(n)]

    # making somethoing to compare against
    # IP.embed() # Debug
    bestPath = cities_int + [cities_int[0]]
    bestScore = fit(bestPath, data)
    print(f"bestPath: {bestPath}, bestScore: {bestScore}")
    current = 0
    # generating all paths
    for perm in itertools.permutations(cities_int, n):
        path = list(perm) + [perm[0]]
        current = fit(path, data)
        if current < bestScore:
            bestScore = current
            bestPath = path
        current = 0
    return bestPath, bestScore

if __name__ == "__main__":

    tmp = exhaustiveSearch(city_data.data_subset(city_data.path_to_datafile, 10))
    # tmp = exhaustiveSearch(city_data.data(city_data.path_to_datafile))

    print(tmp)

   # Running exhaustive search on subset

