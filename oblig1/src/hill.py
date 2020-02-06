import IPython as IP
# --
import pandas as pd
# import numpy as np
import itertools
# import multiprocessing
import time
import matplotlib.pyplot as plt
import random
import data as city_data 


def hill(data, n = 10_000, t = 100):
    cities = data.columns
    n = len(cities)
    cities_int = [i+1 for i in range(n-1)]
    startCity = 0

    # making somethoing to cange
    bestPath = currentPath = [startCity] + cities_int
    bestScore = currentScore = city_data.fit(currentPath, data)
    print(f"startPath: {bestPath}, startScore: {bestScore}")
    startTime = time.time()

    # Running for time t
    while time.time() - startTime < t:
        pick1 = random.randint(1,n-1)
        pick2 = random.randint(1,n-1)

        tmp = currentPath[:]
        tmp[pick1], tmp[pick2] = tmp[pick2], tmp[pick1]
        currentScore = city_data.fit(tmp, data)
        
        if currentScore < bestScore:
            bestPath = tmp[:]
            bestScore = currentScore

    return bestPath, bestScore 



if __name__ == "__main__":
    totTimeStart = time.time()
    timeData = {}
    results = {}
    for i in range(3,11):
        data_subset = city_data.data_subset(city_data.path_to_datafile, i)
        print(f"\nRunning Hill, with i: {i}")
        
        startTime = time.time()
        bestPath, bestScore = hill(data_subset, t = 10)
        endTime = time.time()

        timeData[i] = endTime - startTime
        results[i] = {"time": timeData[i], "bestScore": bestScore, "bestPath": bestPath"}
        print(f"i: {i}, time: {timeData[i]}, bestScore: {bestScore}, bestPath {bestPath}")
    
    totTime = time.time() - totTimeStart
    print(f"\nTotal time: {totTime}")

    # Writing data to file
    city_data.writeResults(results, "hill.json")

    # # plotting time data per size
    # tmp = sorted(timeData.items()) # sorted by key, return a list of tuples
    # x, y = zip(*tmp) # unnpacking the data
    # plt.plot(x, y)
    # # plt.yscale("log")
    # plt.show()
