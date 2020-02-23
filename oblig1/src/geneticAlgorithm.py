import numpy as np
import random
import matplotlib.pyplot as plt

import data as data

import IPython as IP  # - for debug

# Seeds
# np.random.seed(178)
# random.seed(10)


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
    # print(f"pmx:")
    # print(f"  P1: {P1},\n  P2: {P2},\n  c1: {c1}, c2: {c2}")

    # - 1.
    if c1 is None:
        c1 = np.random.randint(0, len(P1))
    if c2 is None:
        c2 = np.random.randint(0, len(P1))  

    crossoverPoint1 = min((c1, c2))
    crossoverPoint2 = max((c1, c2))
    # print(f"  crossoverpoints: {(c1, c2)}")
    segment = P1[crossoverPoint1: crossoverPoint2 + 1]
    # Making the offspringv
    offspring = [None for _ in range(0, len(P1))]
    offspring[crossoverPoint1: crossoverPoint2+1] = segment

    # - 2.
    for P2val in P2[crossoverPoint1: crossoverPoint2 +1]:
        # Looking in the same segment positions in parent 2, select each value that hasn't already been copied to the child.
        if P2val not in offspring:
            val = P2val
            # print(f"    val (from p2Val): {val}, offspring: {offspring}")
            placed = False
            while not placed:
                index_P2val = P2.index(val)
                # print(f"    index_P2val: {index_P2val}")
                v = P1[index_P2val]
                # print(f"    v: {v}")
                index_v_P2 = P2.index(v)
                # print(f"    index_v_P2: {index_v_P2}")
                if index_P2val in range(crossoverPoint1, crossoverPoint2 +1):
                    val = v
                    # print(f"    val: {val}")
                else:
                    offspring[index_P2val] = P2val
                    # print(f"    Inserting {P2val} into offspring: {offspring}")
                    placed = True
                # print("\n")


    # - 6
    # print(f"  Loop ended, and adding rest from p2, offspring: {offspring}")
    for i in range(0,len(P1)):
        if offspring[i] is None:
            offspring[i] = P2[i]
    # print(f"  After adding rest of p2 to offspring: {offspring}")

    return offspring


# def pmx_helper(val, p1, p2, c1, c2, r):
#     print(f"\n    pmx helper:")
#     p2_index_val = p2.index(val)
#     v = p1[p2_index_val]
#     same_val_P2_index = p2.index(v)
#     print(f"      val: {val}, p1: {p1}, p2: {p2}"
#           f", c1: {c1}, c2: {c2}, v: {v}, "
#           f"same_val_P2_index: {same_val_P2_index}, p2_index_val: {p2_index_val}"
#           )
#     if v == None:
#         raise ValueError(" v is none")

#     if same_val_P2_index in range(c1, c2):
#         r = pmx_helper(v, p1, p2, c1, c2, r)
#     else:
#         r = [same_val_P2_index, val]
#         print(f"    pmx helper return r: {r}")
#         return r


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
    mapping = tuple(range(0, len(
        population)))  # Since pop is the whole population. Making a id for each path to be sorted with the corresponding fit value

    # Sort map after rank, with the fit values
    fit_values, mapping = zip(*sorted(
        zip(fit_values[:], mapping)))  # https://stackoverflow.com/questions/5284183/python-sort-list-with-parallel-list

    # Selection
    p = np.array([n - i for i in range(n)])
    p = p / np.sum(p)  # Cumulative prob
    parentsMap = np.random.choice(mapping, p=p, size=int(n * selection_factor), replace=False)
    # print(f"p: {p}") # - Debug

    chosenParents = []
    for i in parentsMap:
        chosenParents.append(population[i])
    return chosenParents

def geneticAlgorithm(popSize, iterations, subsetSize, mutationP, debug = False, verbose = False):
    
    # Assumptions
    # popSize
    assert popSize > 0, "popSize must be larger than 0!"
    #iterations
    assert type(iterations) is int, f"iterations must be int not: {type(iterations)}"
    assert iterations > 0, f"iterations must be larger than 0! Given val {iterations}"
    # subsetSizes
    assert type(subsetSize) is int, f"subsetSize must be int, not: {type(subsetSize)}"
    assert subsetSize > 0, f"subsetSize must be larger than 0! Given val {subsetSize}"
    # mutationP
    try:
        _ = float(mutationP)
    except TypeError:
        print(f"mutationP must be a number. Given val {mutationP}")
    assert mutationP <= 1 and mutationP >= 0, f"mutationP must be between 0 and 1 (must be a probability)! Given val: {mutationP}"
    # Debug
    assert type(debug) is bool or debug == 1 or debug == 0, f"debug must be a bool or similar! Given was: {debug}"

    # Constants
    startCity = 0

    # Storing values
    if debug: print("init values for storing cores")
    minScores = []  # min scores for each iterations
    bestScore = -1  # Best score after all iterations
    bestCandidate = []  # the path with the best score
    if debug: print(f"  minScores: {minScores}, bestScore: {bestScore}, bestPath: {[startCity] + bestCandidate}")

    # Getting cities data
    if debug: print("Getting cities data")
    cities_df = data.data_subset(data.path_to_datafile, sub=subsetSize)
    cities_representation = data.get_representation(cities_df)
    if debug: print(f"  cities_df: {cities_df}, cities_representation: {cities_representation}")

    # Making chromosomes with no bias
    if debug: print("Making chromosomes")
    pop = []
    for i in range(popSize):  # Might be done with a map or numpy to be made faster
        pop.append([i for i in range(1, subsetSize)])  # without 0 since that is the start city
        random.shuffle(pop[i])
    if debug: print(f"  pop: {pop}")

    # Starting GA loop
    if verbose: print(f"Starting algorithm with: \n" +
        f"  popSize: {popSize}, subsetSize: {subsetSize}, mutationP: {mutationP}, iterations: {iterations} \n"
    )
    for mainIndex in range(iterations):
        if debug: print("\n")
        # if iterations % (iterations // 100) == 0: print(f"i: {mainIndex}")
    
        # Population - Evaluate each candidate
        if debug: print("Evaluating each candidate. (assigning fit values). Path is including start city")
        fitValues = []  # Scores of all candidates this iteration
        for candidate in pop:  # Might be improved with map, witch might need to use a lambda function
            fitValues.append(data.fit([startCity] + candidate, cities_df))  # always the same start city
        if debug:
            for i in range(len(pop)):
                print(f"  fitVal: {fitValues[i]}, path: {[startCity] + pop[i]}")

        # Save score for evaluation
        if debug: print("Saving scores for evaluation")
        minScores.append(min(fitValues))
        if min(fitValues) < bestScore or mainIndex == 0:
            """
            If the min of the fit values is smaller then the current smallest replace the bestScore, or if it is the first iteration.
            This can be alleviated by calculating fit values before the loop.
            """
            bestCandidateIndex = fitValues.index(min(fitValues))
            bestScore = min(fitValues)
            bestCandidate = pop[bestCandidateIndex][:]
            if debug: print(f"  bestScore: {bestScore}, bestPath: {[startCity] + bestCandidate}")
            
        # Parent Selection - Rank based Selection
        if debug: print("Parent Selection - Rank bases Selection")
        parents = rankBasedSelection(pop, fitValues)
        if debug:
            for parent in parents:
                print(f" parent: {parent}")

        # Recombination TODO : Ask if some children should survive
        if debug: print("Recombination - Making offsprings off parents")
        """
        Choosing parents to mate. No parents survive, only makes children
        """
        offsprings = []
        numberOfTwinMatings = popSize // 2  # we wish each parent couple to have two children
        numberOfLonelyChildren = popSize % 2  # To keep popSize stable, will max be 1 but keeping it general
        if debug: print(f"  numberOfTwinMatings: {numberOfTwinMatings} , numberOfLonelyChildren: {numberOfLonelyChildren} ")
        for _ in range(numberOfTwinMatings):
            """
            Choosing two parents at random to mate, and generate twins.
            """
            partnerIndex1, partnerIndex2 = np.random.choice(range(0, len(parents)),
                                                            size=2)  # Choosing two parents to mate
            offspring1 = pmx(parents[partnerIndex1], parents[partnerIndex2])
            offspring2 = pmx(parents[partnerIndex2], parents[partnerIndex1])
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        if debug: print(f"  offsprings (only twins): {offsprings}")
        for _ in range(numberOfLonelyChildren):  # Range(0) skips the loop
            """
            Making sure pop size keeps stable
            """
            partnerIndex1, partnerIndex2 = np.random.choice(range(0, len(parents)),
                                                            size=2)  # Choosing two parents to mate, making lonley child
            offspring = pmx(parents[partnerIndex1], parents[partnerIndex2])
            offsprings.append(offspring)
        if debug: print(f"  offsprings: {offsprings}")
        assert len(offsprings) == popSize, f"len(offsprings) != popSize\n  len(offsprings): {offsprings}\n  popSize: {popSize}"

        # Mutation
        if debug: print("Mutation")
        P = np.random.random(size=popSize)
        for j in range(popSize):
            if P[j] < mutationP:
                if debug: print(f"  Mutating: {offsprings[j]}")
                inverstionMutation(offsprings[j])
                if debug: print(f"  Mutated: {offsprings[j]}")
        

        # Survivor selection
        if debug: print(f"New population")
        pop = offsprings[:]
        if debug: print(f"  pop: {pop}")

    # r = { # Best Scores
    #     "minScores" : minScores,
    #     "bestScore": bestScore, 
    #     "bestPath": [startCity] + bestCandidate
    #     }
    r = { # Last scores
        "minScores": minScores,
        "bestScore": min(fitValues),
        "bestPath": [startCity] + pop[fitValues.index(min(fitValues))]
    }
    return r

        
        
        
if __name__ == "__main__":
    """
    PoP is without start city, start city is only added when the score is calculated
    
    """
    # Constants (variables)
    print("Constants")
    popSizes = (10, 15, 100)
    iterations = int(1e3)
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
        fitValues = []  # Scores of all candidates this iteration
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
            bestPath = pop[bestPathIndex][:]

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
            partnerIndex1, partnerIndex2 = np.random.choice(range(0, len(parents)),
                                                            size=2)  # Choosing two parents to mate
            offspring1 = pmx(parents[partnerIndex1], parents[partnerIndex2])
            offspring2 = pmx(parents[partnerIndex2], parents[partnerIndex1])
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        for _ in range(numberOfLonelyChildren):  # Range(0) skips the loop
            """
            Making sure pop size keeps stable
            """
            partnerIndex1, partnerIndex2 = np.random.choice(range(0, len(parents)),
                                                            size=2)  # Choosing two parents to mate, making lonley child
            offspring = pmx(parents[partnerIndex1], parents[partnerIndex2])
            offsprings.append(offspring)

        # Test if len(offsprings) == popSize, to make sure popSize will be stable
        assert len(offsprings) == popSize, \
            f"i: {i}, len(offsprings): {len(offsprings)}, popSize: {popSize}, len(parents): {len(parents)}"
        # - Debug
        print(f"  Offsprings:")
        for offspring in offsprings:
            print(f"    {offspring}")

        # Mutation
        # print("\nMutation")
        # print("  Mutation: (on offspring)")
        P = np.random.random(size=popSize)
        for j in range(popSize):
            if P[j] < mutationP:
                inverstionMutation(offsprings[j])
                # print(f"    {offsprings[j]}")
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
