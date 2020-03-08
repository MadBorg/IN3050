import data
import population
import genotype

"""     General information
Population:
    Representation: Permutation
Recombine:
    Number of parrents per offspring: 2
    PMX
Mutation:

Survivor Selection
    Age biased Survivor Selection (only choosing from children)
    Selecting the bast ones - Deterministic
"""


# Global variables
path_to_datafile = "data//european_cities.csv" # relative path

# -

def fit(model, df):
    x = 0 # summation variable
    for i in range(len(model)-1):
        x += df.iloc[model[i], model[i+1]]
    x += df.iloc[model[-1], model[0]]
    return x



if __name__ == "__main__":
    """
    TODO : Implement for (N-1)!
    """
    # -  Initialise
    # Variables
    start_city = 0
    subset_sizes = [6, 10]

    # Getting data
    cities_data = data.data()
    cities_data.read_csv(path_to_datafile)
    representation = cities_data.get_representation(1, subset_sizes[0])

    # Population
    current_population = population.Population(
        genotype.Genotype,
        representation,
        fit
    )
    # -  Evaluate
    
    # -  Loop
    # -  Parent Selection
    # -  Recombine - Crossover
    # -  Mutate
    # -  Evaluate
    # -  Survivor Selection
    # -  endLoop