# for debug
import IPython as IP
# --
import pandas as pd
import numpy as np
import itertools
import multiprocessing
import time
import matplotlib.pyplot as plt
import data as city_data                                    



def fit(path, data):
    cum = 0;
    for i in range(len(path)-1):
        cum += data.iloc[path[i], path[i+1]]
    # print(f"Fit: path: {path}, score {cum}")
    cum += data.iloc[path[-1], path[0]]
    return cum

def exhaustiveSearch(data):
    cities = data.columns
    n = len(cities)
    cities_int = [i+1 for i in range(n-1)]
    startCity = 0

    # making somethoing to compare against
    # IP.embed() # Debug
    bestPath = [startCity] + cities_int
    bestScore = fit(bestPath, data)
    print(f"startPath: {bestPath}, startScore: {bestScore}")
    current = 0
    # generating all paths
    for perm in itertools.permutations(cities_int, n-1):
        path = [startCity] + list(perm)
        current = fit(path, data)
        if current < bestScore:
            bestScore = current
            bestPath = path
        current = 0
    return bestPath, bestScore

if __name__ == "__main__":

    # # Getting time data from one szie subset
    # n = 4
    # data_subset = city_data.data_subset(city_data.path_to_datafile, n)
    # startTime = time.time()
    # bestPath, bestScore = exhaustiveSearch(data_subset)
    # endTime = time.time()
    # timeData = endTime - startTime
    # print(f"\n\nn: {n}, time: {timeData}, bestScore: {bestScore}, bestPath {bestPath} \n\n")

    # Getting time data from a range of different sizes of subsets
    totTimeStart = time.time()
    timeData = {}
    for i in range(3, 11):
        data_subset = city_data.data_subset(city_data.path_to_datafile, i)
        print(f"\nRunning exhaustiveSearch, with i: {i}")

        startTime = time.time()
        bestPath, bestScore = exhaustiveSearch(data_subset)
        endTime = time.time()
        
        timeData[i] = endTime - startTime
        print(f"i: {i}, time: {timeData[i]}, bestScore: {bestScore}, bestPath {bestPath}")
        
    totTime = time.time() - totTimeStart
    print(f"\nTotal time: {totTime}")
    # plotting time data per size
    tmp = sorted(timeData.items()) # sorted by key, return a list of tuples
    x, y = zip(*tmp) # unnpacking the data
    plt.plot(x, y)
    # plt.yscale("log")
    plt.show()
    

    
    
