import numpy as np
import random

import data as data

import IPython as IP # - for debug
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
    print(f"Genotype: {genotype}") # - Debug
    subsetSize = np.random.randint(2, len(genotype))
    start = np.random.randint(0, len(genotype))
    # subsetSize = start = 3
    print(f"subsetSize: {subsetSize}, start: {start}") # - Debug

    # Getting a subset, treating it as a linked list
    if start + subsetSize > len(genotype):
        print("start + subsetSize > len(genotype)") # - Debug
        subset = genotype[start:]
        subset = subset + genotype[:subsetSize - len(subset)]
    else:
        subset = genotype[start:start+subsetSize]
    print(f"subset: {subset}") # - Debug

    # Scramble
    random.shuffle(subset)
    print(f"subsetShuffle: {subset}") # - Debug

    # genotype = [a,b,c,d,e], Start = 3, subsetSize = 3, subset = [d, e, a]
    # Inserting subset back
    if start + subsetSize > len(genotype): # If overflow
        genotype[start:] = subset[: len( genotype[start:] )]

        genotype[: len(subset[len( genotype[start:] ): ])] = subset[len( genotype[start:] ):]
    else:
        genotype[start:] = subset[:]

    print(genotype) # - Debug
    

def inverstionMutation(genotype):
    """
    Inversion mutation works by randomly selecting two positions in the chromosome and reversing the order in which the values appear between those positions. 
    """
    print(f"Genotype: {genotype}") # - Debug
    subsetSize = np.random.randint(2, len(genotype))
    start = np.random.randint(0, len(genotype))
    # subsetSize = start = 3
    print(f"subsetSize: {subsetSize}, start: {start}") # - Debug

    # Getting a subset, treating it as a linked list
    if start + subsetSize > len(genotype):
        print("start + subsetSize > len(genotype)") # - Debug
        subset = genotype[start:]
        subset = subset + genotype[:subsetSize - len(subset)]
    else:
        subset = genotype[start:start+subsetSize]
    print(f"subset: {subset}") # - Debug

    # Inversion
    subset.reverse()
    print(f"reversedSubset: {subset}") # - Debug

    # genotype = [a,b,c,d,e], Start = 3, subsetSize = 3, subset = [d, e, a]
    # Inserting subset back
    if start + subsetSize > len(genotype): # If overflow
        genotype[start:] = subset[: len( genotype[start:] )]

        genotype[: len(subset[len( genotype[start:] ): ])] = subset[len( genotype[start:] ):]
    else:
        genotype[start:] = subset[:]
    print(genotype) # - Debug


# Crossover - Recombination
def pmx(P1, P2): # - Partially Mapped Crossover

    # - 1. Choose two crossover points at random, and copy the segment between them from the first parent (P1) into the first offspring.
    crossoverPoint1 = np.random.randint(0, len(P1)-1)
    crossoverPoint2 = np.random.randint(crossoverPoint1 + 1, len(P1)) # To make sure the points are valis and avoids index error. Culd also be solved by treating the lists as sircular linked lists.

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
        # vacant = False
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
def rankBasedSelection( initial_order, fit_values, selection_factor = 0.5 ):
    """
    Using initial order of the candidates, witch is the indices of each candidate. 
    """
    n = len(initial_order)
    # Sort pop after rank, with the fit values
    fit_values, sorted_order = zip(*sorted(zip(fit_values, initial_order))) # https://stackoverflow.com/questions/5284183/python-sort-list-with-parallel-list
    for fit, order, in  zip(fit_values, sorted_order):
        print(f"fit: {fit}, order: {order}")
    # Selection
    # selected = []
    # for i in range(n//2):
    #     tmp = random.random()
    #     if tmp > 0.5:
    #         selected.append(sorted_order[i])



    

if __name__ == "__main__":
    # Constants (variables)
    print("Constants")
    popSizes = (10, 15, 100)
    iterations = int( 1e3 )
    subsetSizes = (10, 24)¨
    print(f"    popSizes: {popSizes}, iterations: {iterations}, subsetSizes: {subsetSizes}")

    # Initializtion
    print("\nInitializtion")
    # Iterating through different versions
    # TODO: impement the loops
    popSize = popSizes[0] # TODO : Iterate through popSizes
    subsetSize = subsetSizes[0] # TODO : Iterate through subsetSizes
    print(f"    popsize: {popSize}, subsetSize: {subsetSize}")

    # Geting the data and the representation from data script
    cities_df = data.data_subset(data.path_to_datafile)
    cities_representation = data.get_representation(cities_df)

    # Making chromosomes whith no bias
    pop = [[0]*subsetSize]*popSize
    for chromosome in pop:
        chromosome = [i for i in range(subsetSize)]
        random.shuffle(chromosome)
       

    # Loop
    for i in range(iterations):

    # Population - Evaluate each candidate
        fit_values = []
        for candidate in pop: # Might be improved with map, migth need to use a lambda function
            fit_values.append(data.fit(pop, cities_df))

    # Parent Selection - Rank based Selection


    # Parents

    # Recombination

    # Mutation

    # Offspring

    # Survivor selection

    # End loop

    # Termination



    # IP.embed()


 

