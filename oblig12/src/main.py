import data
import population
import genotype

"""     __General information__
Population:
    Representation: Permutation
Recombine:
    Number of parents must be even
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
    number_of_generations = 100 # Number of iterations, break condition
    start_city = 0 # TODO : implement for start city
    subset_sizes = [6, 10]
    pop_size = 100
    parent_selection_portion = 0.5
    Noffsprings = 2*pop_size
    mutation_p = 0.1

    # Getting data
    cities_data = data.data()
    cities_data.read_csv(path_to_datafile)
    representation = cities_data.get_representation(1, subset_sizes[0])
    subset_data = cities_data.get_subset(subset_sizes[0])

    # Population
    cur_population = population.Population(
        Genotype = genotype.Genotype,
        representation= representation,
        evaluator = fit,
        population_size = pop_size,
        parent_selection_portion = parent_selection_portion,
        number_of_offsprings = Noffsprings,
        mutation_probability = mutation_p
    )
    # -  Evaluate
    cur_population.evaluate_population(df=subset_data, genotype_set="population")
    # -  Loop
    for generation in range():
    # -  Parent Selection
        cur_population.parent_selection()
    # -  Recombine - Crossover
        cur_population.recombination()
    # -  Mutate
        cur_population.mutate()
    # -  Evaluate
        cur_population.evaluate_population(df=subset_data, genotype_set="offsprings")
    # -  Survivor Selection
        cur_population.survivor_selection()
    # -  endLoop