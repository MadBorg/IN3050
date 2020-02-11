import numpy as np
import random

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

    # if start + subsetSize > len(genotype):
    #     i = start
    #     j = 0
    #     while i < len(genotype): # Running until the end of genotype
    #         genotype[i] = subset[j]
    #         i += 1
    #         j += 1
    #     i = 0 # Starting on the start of genotype
    #     while j < subsetSize:
    #         genotype[i] = subset[j]
    #         i += 1
    #         j += 1
    # else:
    #     i = start
    #     j = 0
    #     while j < subsetSize:
    #         genotype[i] = subset[j]
    #         i += 1
    #         j += 1
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
def pmx(): # - Partially Mapped Crossover
    pass

def edgeCrossover():
    pass

def orderCrossover():
    pass

def cycleCrossover():
    pass

# helping functions

#- 
if __name__ == "__main__":
    a = [i for i in range(0,10)]
    IP.embed()


 

