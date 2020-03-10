import population
import genotype
import main
import data

import IPython as ip

df = data.data(main.path_to_datafile)

population_size = 100
subset_size = 6
parent_selection_portion = 0.5
number_of_offsprings = 200
mutation_prob = 0.2
cities_data = data.data()
cities_data.read_csv(main.path_to_datafile)
subset = cities_data.get_subset(subset_size)
representation = cities_data.get_representation(N=subset_size, start=0)

pop = population.Population(
    genotype.Genotype,
    representation,
    main.fit,
    population_size,
    parent_selection_portion,
    number_of_offsprings, 
    mutation_prob
)

def test_population_create_population():
    ind = 0
    counter = 0
    for test_obj in pop._population:
        for comparison_obj in pop._population[ind+1:]:
            if test_obj.r == comparison_obj.r:
                counter +=1
    assert counter != len(pop._population), f"pop: {pop()}"

def test_population_recombination():
    couples = [(i, i+1) for i in range(0,100,2)] # TODO : Implement parent selection

    pop.evaluate_population(subset)
    pop.parent_selection()
    offsprings = pop.recombination()

    assert len(offsprings) == number_of_offsprings, "len offsprings is not the same as number of offsprings" +\
        f"len(offsprings): {len(offsprings)}, number of offsprings: {number_of_offsprings}!"
    for offsping in offsprings:
        assert len(offsping) == subset_size, "len offsping is not equal subset size"

def test_population_mutation():
    """
    Tests that when there was a mutation there is only two differences between in the gene
    """
    pop_before = pop.population[:]
    pop.mutation_probability = 0.5
    pop.mutate()
    pop_after = pop.population[:]
    # ip.embed()
    for gene_before, gene_after in zip(pop_before, pop_after):
        if not gene_before.r == gene_after.r:
            counter = 0
            for allele_before, allele_after in zip(gene_before.r, pop_after.r):
                counter += allele_before == allele_after
            if not counter == 2:
                raise ValueError(f"population_mutation: before: {gene_before}, after: {gene_after}")

if __name__ == "__main__":
    test_population_create_population()
    test_population_recombination()
    test_population_mutation()