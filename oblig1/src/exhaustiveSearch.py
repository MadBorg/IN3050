# for debug
import IPython as IP
# --
import pandas as pd
import numpy as np

import data as city_data

def exhaustiveSearch(data, seq=[]):
    n, m = data.shape

    for i in range(m):
        # print(f"m = {m}")
        col = data.columns[i]
        seq.append(col)
        exhaustiveSearch(
            data.drop( col, axis=1),
            seq,
            )
        seq = []
    print(seq)
    

if __name__ == "__main__":
    subset_cities_df = city_data.data_subset(city_data.path_to_datafile, sub=3)
    exhaustiveSearch(subset_cities_df)

   # Running exhaustive search on subset

