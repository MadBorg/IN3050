import numpy as np
import random
import matplotlib.pyplot as plt

import data as data

import IPython as IP  # - for debug

# Seeds
np.random.seed(178)
random.seed(10)


# mutation
def swapMutation(genotype):
    """
    Swap Mutation: Two positions (genes) in the chromosome are selected at random and their allele values swapped.
    """
    locus1 = locus2 = 0
    while locus1 == locus2:  # - To ensure they are not the same index
        locus1, locus2 = loci = np.random.randint(0, len(genotype), size=2)  # - plural of locus
    tmp = genotype[locus1]
    genotype[locus1] = genotype[locus2]
    genotype[locus2] = tmp
    # return genotype - Not needed since it changes the original list


def insertMutation(genotype):
    """
    Insert Mutation: Two alleles are selected at random and the second moved next to the first, shuffling along the others to make room. 
    """
    allele1 = allele2 = 0
    while allele1 == allele2:  # - To ensure they are not the same value
        allele1, allele2 = alleles = np.random.randint(min(genotype), max(genotype),
                                                       size=2)  # two values chosen at random, assuming all the allelespace is between max gen and min gen

    # allele to locus - Quite sure finding alleles is the same as choosing loci. But i did as in the description
    locus1 = genotype.index(allele1)
    # locus2 = genotype.index(allele2) # - Did not need this

    # Doing the crossover
    genotype.remove(allele2)
    genotype.insert(locus1, allele2)


def scrambleMutation(genotype):
    """
    Here the entire chromosome, or some randomly chosen subset of values within it, have their positions scrambled.
    """
    # print(f"\nStart - Genotype: {genotype}") # - Debug
    subsetSize = np.random.randint(2, len(genotype))
    start = np.random.randint(0, len(genotype))
    # subsetSize = start = 3
    # print(f"subsetSize: {subsetSize}, start: {start}") # - Debug

    # Getting a subset, treating it as a linked list
    if start + subsetSize > len(genotype):
        # print("start + subsetSize > len(genotype)") # - Debug
        subset = genotype[start:]
        subset = subset + genotype[:subsetSize - len(subset)]
    else:
        subset = genotype[start:start + subsetSize]
    # print(f"subset: {subset}") # - Debug

    # Scramble
    random.shuffle(subset)
    # print(f"subsetShuffle: {subset}") # - Debug

    # genotype = [a,b,c,d,e], Start = 3, subsetSize = 3, subset = [d, e, a]
    # Inserting subset back
    if start + subsetSize > len(genotype):  # If overflow
        genotype[start:] = subset[: len(genotype[start:])]

        genotype[: len(subset[len(genotype[start:]):])] = subset[len(genotype[start:]):]
    else:
        genotype[start:start + subsetSize] = subset[:]

    # print(f"End of genotype: {genotype}") # - Debug


def inverstionMutation(genotype):
    """
    Inversion mutation works by randomly selecting two positions in the chromosome and reversing the order in which the values appear between those positions. 
    """
    # print(f"\nStart - Genotype: {genotype}") # - Debug
    subsetSize = np.random.randint(2, len(genotype))
    start = np.random.randint(0, len(genotype))
    # subsetSize = start = 3
    # print(f"subsetSize: {subsetSize}, start: {start}") # - Debug

    # Getting a subset, treating it as a linked list
    if start + subsetSize > len(genotype):
        # print("start + subsetSize > len(genotype)") # - Debug
        subset = genotype[start:]
        subset = subset + genotype[:subsetSize - len(subset)]
    else:
        subset = genotype[start:start + subsetSize]
    # print(f"subset: {subset}") # - Debug

    # Inversion
    subset.reverse()
    # print(f"reversedSubset: {subset}") # - Debug

    # genotype = [a,b,c,d,e], Start = 3, subsetSize = 3, subset = [d, e, a]
    # Inserting subset back
    if start + subsetSize > len(genotype):  # If overflow
        genotype[start:] = subset[: len(genotype[start:])]

        genotype[: len(subset[len(genotype[start:]):])] = subset[len(genotype[start:]):]
    else:
        genotype[start:start + subsetSize] = subset[:]
    # print(f"End of genotype: {genotype}") # - Debug


# Crossover - Recombination
def pmx(P1, P2, c1=None, c2=None):  # - Partially Mapped Crossover

    # - 1. Choose two crossover points at random, and copy the segment between them from the first parent (P1) into the first offspring.
    if c1 is None:
        crossoverPoint1 = np.random.randint(0, len(P1) - 1)
    else:
        crossoverPoint1 = c1
    if c2 is None:
        crossoverPoint2 = np.random.randint(crossoverPoint1 + 1, len(
            P1))  # To make sure the points are valis and avoids index error. Culd also be solved by treating the lists as sircular linked lists.
    else:
        crossoverPoint2 = c2
    # crossoverPoint1 = 3 # - Debug
    # crossoverPoint2 = 7 # - Debug

    segment = P1[crossoverPoint1: crossoverPoint2]
    offspring = [None for i in range(0, len(P1))]
    offspring[crossoverPoint1: crossoverPoint2] = segment

    # print(f"P1: {P1}, \nP2: {P2}, \noffspring: {offspring}") # - Debug
    # - 2. Starting from the first crossover point look for elements in that segment of the second parent (P2) that have not been copied.
    for i in P2[crossoverPoint1: crossoverPoint2]:
        # print(f"i: {i}") # - Debug
        # - 3. For each of these (say i), look in the offspring to see what element (say j) has been copied in its place from P1.
        if i not in offspring:
            j = offspring[P2.index(i)]
            # print(f"j: {j}") # - Debug

            # - 4. Place i into the position occupied by j in P2, since we know that we will not be putting j there (as we already have it in our string).
            # IP.embed()
            if offspring[P2.index(j)] is None:
                offspring[P2.index(j)] = i
            # - 5. If the place occupied by j in P2 has already been filled in the offspring by an element k, put i in the position occupied by k in P2.
            else:
                k = offspring[P2.index(j)]
                # print(f"k: {k}")
                offspring[P2.index(k)] = i
        # print(f"offspring: {offspring}") # - Debug 

    for i in range(len(offspring)):
        if offspring[i] == None:
            offspring[i] = P2[i]

    return offspring


def edgeCrossover():
    pass


def orderCrossover():
    pass


def cycleCrossover():
    pass


# Parent selection
def rankBasedSelection(population, fit_values, selection_factor=0.5):
    """
    Making a mapping of the population. So that we can get the right index out without changing order in population, and dont have to move the whole population around
    """
    n = len(population)
    mapping = tuple(range(0, len(population)))  # Since pop is the whole population. Making a id for each path to be sorted with the corresponding fit value

    # Sort map after rank, with the fit values
    fit_values, mapping = zip(*sorted(zip(fit_values[:], mapping)))  # https://stackoverflow.com/questions/5284183/python-sort-list-with-parallel-list

    # Selection
    p = np.array([n-i for i in range(n)])
    p = p / np.sum(p)  # Cumulative prob
    parentsMap = np.random.choice(mapping, p=p, size=int(n * selection_factor), replace=False)
    # print(f"p: {p}") # - Debug

    chosenParents = []
    for i in parentsMap:
        chosenParents.append(population[i])
    return chosenParents


if __name__ == "__main__":
    """
    PoP is without start city, start city is only added when the score is calculated
    
    """
    # Constants (variables)
    print("Constants")
    popSizes = (4, 15, 100)
    iterations = int(1e1)
    subsetSizes = (10, 24)
    startCity = 0
    mutationP = 0.1
    print(f"    popSizes: {popSizes}, iterations: {iterations}, subsetSizes: {subsetSizes}, startCity: {startCity}")

    # Constants for evaluation
    minScores = []  # min scores for each iterations
    bestScore = -1  # Best score after all iterations
    bestPath = []  # the path with the best score

    # Initialization
    print("\nInitialization")
    # Iterating through different versions
    # TODO: implement the loops
    popSize = popSizes[0]  # TODO : Iterate through popSizes
    subsetSize = subsetSizes[0]  # TODO : Iterate through subsetSizes
    print(f"    pop Size: {popSize}, subset Size: {subsetSize}")

    # Getting the data and the representation from data script
    print("\nGetting the data")
    cities_df = data.data_subset(data.path_to_datafile, sub=subsetSize)
    cities_representation = data.get_representation(cities_df)
    print(f"    cities: {cities_df.columns}, cities_representation: {cities_representation}")

    # Making chromosomes with no bias
    print("\nMaking chromosomes")
    pop = []
    for i in range(popSize):  # Might be done with a map or numpy to be made faster
        pop.append([i for i in range(1, subsetSize)])  # without 0 since that is the start city
        random.shuffle(pop[i])
    print(f"pop: {pop}")

    # Loop
    print(f"\nStart Loop, with iterations: {iterations}")

    for mainIndex in range(iterations):
        print(f"i: {mainIndex}")
        # Population - Evaluate each candidate
        fitValues = [] # Scores of all candidates this iteration
        for candidate in pop:  # Might be improved with map, witch might need to use a lambda function
            fitValues.append(data.fit([startCity] + candidate, cities_df))  # always the same start city
        # - Debug
        print("  pop:")
        for j in range(len(pop)):
            print(f"    {pop[j]}, fitVal: {fitValues[j]}")

        # Save score for evaluation
        minScores.append(min(fitValues))
        if min(fitValues) < bestScore or i == 0:
            """
            If the min of the fit values is smaller then the current smallest replace the bestScore, or if it is the first iteration.
            This can be alleviated by calculating fit values before the loop.
            """
            bestPathIndex = fitValues.index(min(fitValues))
            bestScore = min(fitValues)
            bestPath = pop[ bestPathIndex ][:]

        # Parent Selection - Rank based Selection
        parents = rankBasedSelection(pop, fitValues)

        # - Debug
        print("  Parents:")
        for parent in parents:
            print(f"    {parent}")

        # Recombination TODO : Ask if some children should survive
        """
        Choosing parents to mate. No parents survive, only makes children
        """
        offsprings = []
        numberOfTwinMatings = popSize // 2  # we wish each parent couple to have two children
        numberOfLonelyChildren = popSize % 2  # To keep popSize stable, will max be 1 but keeping it general
        for _ in range(numberOfTwinMatings):
            """
            Choosing two parents at random to mate, and generate twins.
            """
            partnerIndex1, partnerIndex2 = np.random.choice(range(0, len(parents)), size=2) # Choosing two parents to mate
            offspring1 = pmx( parents[partnerIndex1], parents[partnerIndex2] )
            offspring2 = pmx( parents[partnerIndex2], parents[partnerIndex1] )
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        for _ in range(numberOfLonelyChildren):  # Range(0) skips the loop
            """
            Making sure pop size keeps stable
            """
            partnerIndex1, partnerIndex2 = np.random.choice(range(0, len(parents)), size=2)  # Choosing two parents to mate, making lonley child
            offspring = pmx( parents[partnerIndex1], parents[partnerIndex2] )
            offsprings.append(offspring)
        # Test if len(offsprings) == popSize, to make sure popSize will be stable
        assert len(offsprings) == popSize,  \
            f"i: {i}, len(offsprings): {len(offsprings)}, popSize: {popSize}, len(parents): {len(parents)}"
        # - Debug
        print(f"  Offsprings:")
        for offspring in offsprings:
            print(f"    {offspring}")

        # Mutation
        # print("\nMutation")
        print("  Mutation: (on offspring)")
        P = np.random.random(size=popSize)
        for j in range(popSize):
            if P[j] < mutationP:
                inverstionMutation(offsprings[j])
                print(f"    {offsprings[j]}")
        # - Debug


        # Offspring

        # Survivor selection
        pop = offsprings[:]


    # Show progress:
    # print(f"i: {i}")

    # Test
    # if len(offsprings) != popSize:
    #     print(f"len offsprings ({len(offsprings)}) is not equal popSize ({popSize})")
    #     raise ValueError()
    # for offspring in offsprings:
    #     if len(offspring) != subsetSize:
    #         print(f"len offspring ({len(offspring)}) is not equal subsetSize ({subsetSize})")
    #         raise ValueError()
    # End loop

    # Termination
    print(bestScore)

    # print(scores[:10])
    plt.plot(np.arange(iterations), minScores)
    plt.show()

    # IP.embed()
