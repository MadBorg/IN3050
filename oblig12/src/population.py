import numpy as np
import genotype

class Population:
    def __init__(self, Genotype, representation, evaluator, population_size, parent_selection_portion):
        """
        Input:
        -----
        Genotype: Genotype (class)
            From what to make the population
        representation: iterable
            an example on how a representation looks, to make a randomized similar population
        evaluator: function
            a function for calculating the fitness values
        population_size: int
            Determins how big the population should be
        parent_selection_portion: float
            Determiens witch portion of the population is going to be used ot make offsprings 
        """
        # Asserts
        assert type(population_size) is int, "Population size must be int"

        # Storing
        self.genotype = genotype
        self.population_size = population_size

        # Creating population
        pop = [] 
        for i in range(population_size):
            pop.append(
                Genotype(
                    representation,
                    evaluator
                )
            )
        self._population = pop
    
        # 

    def __call__(self):
        return self._population
    
    # - Evaluate
    def evaluate_population(self, df):
        # Using the fact that list are order specific
        scores = []
        for genotype in self():
            scores.append(
                genotype.calculate_score(df)
            )
        self._scores = scores
        return scores


    # - Parent Selection 
    def ranked_based_selection(self):
        scores = self._scores[:]
        # Making a mapping of the population of indices to the population, this is to be sorted and used for selecting parents
        mapping = [i for i in range(0, len(self._population))] # map of population

        # Sorting after scores, rank is the position in the list
        scores, mapping = zip(*sorted(zip(scores, mapping)))

        # TODO : Check for DESC or ASC, good night


    # - Recombination (Crossover)
    def recombination(self, couples, N, crossover_method="PMX", children_per_couple=2):
        """
        Input:
        ----
        N: int
            Number of offsprings
        couples:
            pairs of preselected parents
        
        Info:
        -----
        Recommended:
            N % (couples * children per couple) == 0
            children_per_couple = 2 or a multiple of 2
        """
        # Assumptions
        assert N > self.population_size, "recombination: N must be equal or bigger than population size"
        if (len(couples) * children_per_couple) % N == 0:
            print(f"WARNING! - N % (len(couples) * children_per_couple) != 0, its {(len(couples) * children_per_couple)}")

        # -
        if crossover_method == "PMX":
            method = self.pmx
        else:
            raise NameError("Crossover method does not exist")

        offsprings = []
        for parents in couples:
            c = 0 # counter for choosing parents
            for i in range(children_per_couple):
                offsprings.append(
                    method(
                        parents[c%children_per_couple], 
                        parents[(c+1)%children_per_couple]
                        ) # With modulo so that the parents will alternate
                )
                c += 1
    
        self.offsprings = offsprings
        return offsprings

    # PMX
    def pmx(self, P1, P2, c1=None, c2=None):
        # print(f"pmx:")
        # print(f"  P1: {P1},\n  P2: {P2},\n  c1: {c1}, c2: {c2}")

        # - 1.
        if c1 is None:
            c1 = np.random.randint(0, P1)
        if c2 is None:
            c2 = np.random.randint(0, len(P1))  

        crossoverPoint1 = min((c1, c2))
        crossoverPoint2 = max((c1, c2))
        # print(f"  crossoverpoints: {(c1, c2)}")
        segment = P1[crossoverPoint1: crossoverPoint2 + 1]
        # Making the offspring
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
    # Edge crossover
    # Order crossover
    # Cycle Crossover

    # - Survivor Selection Mechanism (Replacement)


    # Crossover
    # evaluation
    # Parent selection
    def print_pop(self):
        print("Population:")
        for obj in self._population:
            print(f"{obj()}")

