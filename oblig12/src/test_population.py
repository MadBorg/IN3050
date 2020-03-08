import population
import genotype
import main
import data

df = data.data(main.path_to_datafile)

N = 100
subset_size = 6
cities_data = data.data()
cities_data.read_csv(main.path_to_datafile)
representation = cities_data.get_representation(N=subset_size, start=0)

pop = population.Population(
    genotype.Genotype,
    representation,
    main.fit,
    N
)

def test_population_create_population():
    ind = 0
    counter = 0
    for test_obj in pop._population:
        for comparison_obj in pop._population[ind+1:]:
            if test_obj.r == comparison_obj.r:
                counter +=1
    assert counter != len(pop._population), f"pop: {pop()}"

# def test_population_recombination():
#     number_of_offsprings = 200
#     couples = [(i, i+1) for i in range(0,100,2)] # TODO : Implement parent selection
#     offsprings = pop.recombination(
#         couples,
#         number_of_offsprings
#     )

#     assert len(offsprings) == number_of_offsprings, "len offsprings is not the same as number of offsprings" +\
#         f"len(offsprings): {len(offsprings)}, number of offsprings: {number_of_offsprings}!"
#     for offsping in offsprings:
#         assert len(offsping) == subset_size, "len offsping is not equal subset size"


if __name__ == "__main__":
    test_population_create_population()
    # test_population_recombination()