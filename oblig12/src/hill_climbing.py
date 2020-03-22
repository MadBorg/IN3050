import src.data as data
import numpy as np
import random

def hill(data, fit, numberOfIterations = 10_000):
    cities = data.columns
    n = len(cities)
    cities_int = [i+1 for i in range(n-1)]
    random.shuffle(cities_int)
    startCity = 0

    bestPath = currentPath = [startCity] + cities_int
    bestScore = currentScore = fit(currentPath, data)

    pick1 = np.random.randint(1, n, size=numberOfIterations)
    pick2 = np.random.randint(1, n, size=numberOfIterations)
    for i in range(numberOfIterations):
        p1 = pick1[i]
        p2 = pick2[i]

        tmp = currentPath[:]
        tmp[p1], tmp[p2] = tmp[p2], tmp[p1]
        currentScore = fit(tmp, data)
        
        if currentScore < bestScore:
            bestPath = tmp[:]
            bestScore = currentScore

    return bestPath, bestScore 