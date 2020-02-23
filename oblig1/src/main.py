import time
import matplotlib.pyplot as plt
import numpy as np

import geneticAlgorithm as ga
import exhaustiveSearch as es
import hill
import data

import IPython as ip

if __name__ == "__main__":
    # exhaustive search
    es_time_data = {} # For timing each iteration
    exhaustive_search_data = {} # For staring the best paths and socres
    for i in range(6,11): # iterating trough subset sizes
        start_time = time.time()
        bestPath, bestScore = es.exhaustiveSearch(i) # Running algorithm
        end_time = time.time()
        # Storing data
        exhaustive_search_data[i] = (bestScore, bestPath)
        es_time_data[i] = end_time - start_time
    print(es_time_data)
    
    # Hill Climbing
    bestScore, worstScore = -1, -1
    numCities_list = list(range(2,11)) + [24]
    runs = 20
    hill_scores = {}
    bestScores =  {}
    worstScores =  {}
    sdScores =  {}
    times_run =  {}
    means = {}

    for numCities in  numCities_list:
        scores = []
        times = []
        for _ in range(runs):

            data_subset = data.data_subset(data.path_to_datafile, numCities)
            start_time = time.time()
            path, score = hill.hill(data_subset, 1000)
            end_time = time.time()

            scores.append(score)
            times.append(end_time - start_time)

        # Best
        bestScores[numCities] = min(scores)
        # Worst
        worstScores[numCities] = max(scores)
        # Mean
        # ip.embed()
        mean = sum(scores) / len(scores)
        means[numCities] = mean
        # Sd
        s = 0
        for score in scores:
            s += (score - mean)**2
        s = (s/len(scores))**(1/2)
        sdScores[numCities] = s
        # Time
        mean_time = sum(times) / len(times)
        times_run[numCities] =  mean_time

    # Report
    for numCities in numCities_list:
        print(
            f"numCities: {numCities}\n" +
            f"  Best: {bestScores[numCities]},\n" +
            f"  Worst: {worstScores[numCities]}, \n" +
            f"  SD: {sdScores[numCities]}"
        )

    # Genetic algorithm
    print("Genetic Algorithm")
    numCities_list = [6,10, 24]
    runs = 20
    iterations = 100
    mutation_prob = 0.05
    pop_size = 10
    hill_scores = {}
    bestScores =  {}
    worstScores =  {}
    sdScores =  {}
    times_run =  {}
    means = {}
    average_scores_run = {}

    # r = ga.geneticAlgorithm(
    #     10,
    #     10000,
    #     10,
    #     0.05,
    #     debug=False
    # )

    for numCities in  numCities_list:
        local_min_scores = []
        scores = []
        times = []
        for _ in range(runs):
            data_subset = data.data_subset(data.path_to_datafile, numCities)
            start_time = time.time()
            r = ga.geneticAlgorithm(
                                        pop_size,
                                        iterations,
                                        numCities,
                                        mutation_prob,
                                        debug=False
                                    )
            score = r["bestScore"]
            path = r["bestPath"]
            minScores =r["minScores"]

            end_time = time.time()

            scores.append(score)
            times.append(end_time - start_time)
            local_min_scores.append(minScores)
        
        tmp_averageScores = []
        # ip.embed()
        try:
            for element in zip(*local_min_scores):
                tmp_averageScores.append(
                    sum(element) / len(element)
                )
        except TypeError:
            ip.embed()
        
        average_scores_run[numCities] = tmp_averageScores


        # Best
        bestScores[numCities] = min(scores)
        # Worst
        worstScores[numCities] = max(scores)
        # Mean
        mean = sum(scores) / len(scores)
        means[numCities] = mean
        # Sd
        s = 0
        for score in scores:
            s += (score - mean)**2
        s = (s/len(scores))**(1/2)
        sdScores[numCities] = s
        # Time
        mean_time = sum(times) / len(times)
        times_run[numCities] =  mean_time
        # minScores


    for numCities in numCities_list:
        print(
            f"numCities: {numCities}\n" +
            f"  Best: {bestScores[numCities]},\n" +
            f"  Worst: {worstScores[numCities]}, \n" +
            f"  SD: {sdScores[numCities]},\n" +
            f"  Time: {times_run[numCities]}"
        )
    # ip.embed()
    fig = plt.figure()
    x = np.arange(iterations)
    for numCities in numCities_list:
        y = average_scores_run[numCities]
        plt.plot(x, y)
    plt.show()
    




    
    




