import numpy as np
import random
import matplotlib.pyplot as plt

import data as data

import IPython as IP # - for debug

# Seeds
np.random.seed(178)
random.seed(10)

# mutation
def swapMutation(genotype):
    """
    Swap Mutation: Two positions (genes) in the chromosome are selected at random and their allele values swapped.
    """
    locus1 = locus2 = 0
    while locus1 == locus2: # - To ensure they are not the same index
        locus1, locus2 = loci = np.random.randint(0, len(genotype), size=2) # - plural of locus
    tmp = genotype[locus1]
    genotype[locus1] = genotype[locus2]
    genotype[locus2] = tmp
    # return genotype - Not needed since it changes the original list


def insertMutation(genotype):
    """
    Insert Mutation: Two alleles are selected at random and the second moved next to the first, shuffling along the others to make room. 
    """
    allele1 = allele2 =  0
    while allele1 == allele2: # - To ensure they are not the same value
        allele1, allele2 = alleles = np.random.randint(min(genotype), max(genotype), size=2) # two values chosen at random, assuming all the allelespace is between max gen and min gen
    
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
        subset = genotype[start:start+subsetSize]
    # print(f"subset: {subset}") # - Debug

    # Scramble
    random.shuffle(subset)
    # print(f"subsetShuffle: {subset}") # - Debug

    # genotype = [a,b,c,d,e], Start = 3, subsetSize = 3, subset = [d, e, a]
    # Inserting subset back
    if start + subsetSize > len(genotype): # If overflow
        genotype[start:] = subset[: len( genotype[start:] )]

        genotype[: len(subset[len( genotype[start:] ): ])] = subset[len( genotype[start:] ):]
    else:
        genotype[start:start+subsetSize] = subset[:]

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
        subset = genotype[start:start+subsetSize]
    # print(f"subset: {subset}") # - Debug

    # Inversion
    subset.reverse()
    # print(f"reversedSubset: {subset}") # - Debug

    # genotype = [a,b,c,d,e], Start = 3, subsetSize = 3, subset = [d, e, a]
    # Inserting subset back
    if start + subsetSize > len(genotype): # If overflow
        genotype[start:] = subset[: len( genotype[start:] )]

        genotype[: len(subset[len( genotype[start:] ): ])] = subset[len( genotype[start:] ):]
    else:
        genotype[start:start+subsetSize] = subset[:]
    # print(f"End of genotype: {genotype}") # - Debug



# Crossover - Recombination
def pmx(P1, P2, c1 = None, c2 = None): # - Partially Mapped Crossover

    # - 1. Choose two crossover points at random, and copy the segment between them from the first parent (P1) into the first offspring.
    if c1 is None:
        crossoverPoint1 = np.random.randint(0, len(P1)-1)
    else:
        crossoverPoint1 = c1
    if c2 is None:
        crossoverPoint2 = np.random.randint(crossoverPoint1 + 1, len(P1)) # To make sure the points are valis and avoids index error. Culd also be solved by treating the lists as sircular linked lists.
    else:
        crossoverPoint2 = c2
    # crossoverPoint1 = 3 # - Debug
    # crossoverPoint2 = 7 # - Debug

    segment = P1[crossoverPoint1 : crossoverPoint2]
    offspring = [None for i in range(0, len(P1))]
    offspring[crossoverPoint1 : crossoverPoint2] = segment
    
    # - 2. Starting from the first crossover point look for elements in that segment of the second parent (P2) that have not been copied.
    # - 3. For each of these (say i), look in the offspring to see what element (say j) has been copied in its place from P1.
    # - 4. Place i into the position occupied by j in P2, since we know that we will not be putting j there (as we already have it in our string).
    # - 5. If the place occupied by j in P2 has already been filled in the offspring by an element k, put i in the position occupied by k in P2.
    for i in range(crossoverPoint1, crossoverPoint2):
        val = P2[i]
        if not val in offspring:
            j = _pmxIndex(i, P2, offspring)
            offspring[j] = val

    # Filling the rest of  the offspring with the rest from P2
    for i in range(len(offspring)):
        if offspring[i] is None:
            offspring[i] = P2[i]
    
    return offspring



def _pmxIndex(i, P, offspring): # lookIn = 0: Look in offspring, lookIn = 1: Look in P
    # currentVal = P[i]
    if offspring[i] == None:
        # offspring[i] = currentVal
        return i
    else:
        i = P.index(offspring[i])
        return _pmxIndex(i, P, offspring)
    return i

            
def edgeCrossover():
    pass

def orderCrossover():
    pass

def cycleCrossover():
    pass

# Parent selection
def rankBasedSelection( pop, fit_values, selection_factor = 0.5 ):
    """
    Using initial order of the candidates, witch is the indices of each candidate. 
    """
    n = len(pop)
    order = [i for i in range(n)] # Using order instead of moving pop around. # nott  needed

    # Sort pop after rank, with the fit values
    fit_values, sorted_order = zip(*sorted(zip(fit_values, order))) # https://stackoverflow.com/questions/5284183/python-sort-list-with-parallel-list
    # # - Debug
    # for fit, i, in  zip(fit_values, sorted_order):
    #     print(f"fit: {fit}, i: {i}, chromosome: {pop[i]}")

    # Selection
    p = np.array([i for i in range(n)])
    p = p / np.sum(p) # Cumulative prob
    parentsOrder = np.random.choice(sorted_order, p=p, size = int(n * selection_factor), replace = False)
    # print(f"p: {p}") # - Debug
    parents = []
    for i in parentsOrder:
        parents.append(pop[i])
    return parents


if __name__ == "__main__":
    # Constants (variables)
    print("Constants")
    popSizes = (4, 15, 100)
    iterations = int( 1e3 )
    subsetSizes = (5, 24)
    startCity = 0
    mutationP = 0.01
    print(f"    popSizes: {popSizes}, iterations: {iterations}, subsetSizes: {subsetSizes}, startCity: {startCity}")

    # Constants for evaluation
    scores = []
    best = [-1]

    # Initializtion
    print("\nInitializtion")
    # Iterating through different versions
    # TODO: impement the loops
    popSize = popSizes[0] # TODO : Iterate through popSizes
    subsetSize = subsetSizes[0] # TODO : Iterate through subsetSizes
    print(f"    popsize: {popSize}, subsetSize: {subsetSize}")

    # Geting the data and the representation from data script
    print("\nGetting the data")
    cities_df = data.data_subset(data.path_to_datafile, sub=subsetSize)
    cities_representation = data.get_representation(cities_df)
    print(f"    cities: {cities_df.columns}, cities_representation: {cities_representation}")

    # Making chromosomes whith no bias
    print("\nMaking chromosomes")
    pop = []
    for i in range(popSize):
        pop.append([i for i in range(1, subsetSize)])
        random.shuffle(pop[i])
    print(f"pop: {pop}")
       

    # Loop
    print(f"\nStart Loop, with iterations: {iterations}")
    for i in range(iterations):

    # Population - Evaluate each candidate
        fitValues = []
        for candidate in pop: # Might be improved with map, wich might need to use a lambda function
            fitValues.append(data.fit([startCity] + candidate, cities_df)) # always the same start city

    # Save score for evaluation
        scores.append(min(fitValues[:]))
        if min(fitValues) < best[0] or best[0] == -1:
            i = fitValues.index(min(fitValues))
            best = [min(fitValues), pop[i]]
    # Parent Selection - Rank based Selection
        parents = rankBasedSelection(pop, fitValues[:]) 
        
    # Recombination
        # print("\nRecombination")
        offsprings = []
        while len(offsprings) < popSize:
            partner1, partner2 = np.random.randint(0, len(parents), size=2)
            offsprings.append(pmx(parents[ partner1 ], parents[ partner2 ]))
        
        # print(f"offspring: {offsprings}")
        
    # Mutation
        # print("\nMutation")
        P = np.random.random(size=popSize)
        for i in range(popSize):
            if P[i] < mutationP:
                inverstionMutation(offsprings[i])
        # print(f"\nMutated: {offsprings}")
    # Offspring

    # Survivor selection
        pop = offsprings[:]
    # Show progress:
        # print(f"i: {i}")

    # Test
        if len(offsprings) != popSize:
            print(f"len offsprings ({len(offsprings)}) is not equal popSize ({popSize})")
            raise ValueError()
        for offspring in offsprings:
            if len(offspring) != subsetSize-1:
                print(f"len offspring ({len(offspring)}) is not equal subsetSize ({subsetSize})")
                raise ValueError()
    # End loop

    # Termination
    print(best)

    print(scores[:10])
    plt.plot(np.arange(iterations), scores)
    plt.show()


    # IP.embed()


 

