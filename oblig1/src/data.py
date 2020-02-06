import pandas as pd
import IPython as IP
import datetime
import json

# path_to_datafile = "assignment01\\european_cities.csv"
path_to_datafile = "oblig1\\assignment01\\european_cities.csv"

# data = pd.read_csv(path_to_datafile, sep=";")


def data_subset(path, sub = 6):
    data = pd.read_csv(path, sep=";")
    return data.iloc[:sub, :sub]

def data(path):
    return  pd.read_csv(path, sep=";")

def fit(path, data):
    cum = 0;
    for i in range(len(path)-1):
        cum += data.iloc[path[i], path[i+1]]
    # print(f"Fit: path: {path}, score {cum}")
    cum += data.iloc[path[-1], path[0]]
    return cum

def writeResults(results, fileName):
    data = {str(datetime.datetime.now()): results}
    with open(fileName, "w+") as outFile:
        json.dump(data, outFile, indent=4)




if __name__ == "__main__":
    df = data_subset(path_to_datafile, 3)
    IP.embed()
