import pandas as pd

def data_subset(path, N):
    """
    Return:
    -----
    Returning a subset of the data matrix

    Input:
    -----
    N: int
        Subset_size
    Path: string
        Path to the data file

    How:
    -----
    Reading a csv file to a pandas dataframe
    """
    data = pd.read_csv(path, sep=";")
    return data.iloc[:N, :N]

class data:
    def __init__(self, path_to_csv=None):
        if path_to_csv:
            self.read_csv(path_to_csv)

    def read_csv(self, path, seperator=";"):
        self.path = path
        self.df = pd.read_csv(self.path, sep=seperator)
    
    def get_subset(self, N):
        """
        Input:
        -----
        N: int
            Subset_size
        """
        return self.df.iloc[:N, :N]

    def get_representation(self, N, start=0):
        """
        Input:
        -----
        N: int
            Representation Size
        Return: list
        -----
        Simple representation list of integers from 0 to (not including) N
        """
        return [i for i in range(start, N)]
