import itertools

import src.data as data

def fit(model, df):
    # ip.embed()
    x = 0 # summation variable
    for i in range(len(model)-1):
        x += df.iloc[model[i], model[i+1]]
    x += df.iloc[model[-1], model[0]]
    return x

def exhaustiveSearchEngine(data):
    cities = data.columns
    n = len(cities)
    cities_int = [i+1 for i in range(n-1)]
    startCity = 0

    # making somethoing to compare against
    # IP.embed() # Debug
    bestPath = [startCity] + cities_int
    bestScore = fit(bestPath, data)
    # print(f"startPath: {bestPath}, startScore: {bestScore}")
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

def exhaustiveSearch(subset_size):
    city_data = data.data(path_to_csv = "data//european_cities.csv")
    data_subset = city_data.get_subset(subset_size)
    r = (bestPath, bestScore) = exhaustiveSearchEngine(data_subset)
    return r

if __name__ == "__main__":
    pass
