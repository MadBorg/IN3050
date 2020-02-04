import pandas as pd
import IPython as IP

path_to_datafile = "C:\\Projects\\IN3050\\oblig1\\assignment01\\european_cities.csv"
# data = pd.read_csv(path_to_datafile, sep=";")


def data_subset(path, sub = 6):
    data = pd.read_csv(path, sep=";")
    return data.iloc[:sub, :sub]

def data(path):
    return  pd.read_csv(path, sep=";")

if __name__ == "__main__":
    IP.embed()