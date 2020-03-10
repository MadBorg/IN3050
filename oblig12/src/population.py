import numpy as np
import IPython as ip

import genotype

class Population:
    def __init__(self, Genotype, representation, evaluator, population_size, parent_selection_portion, number_of_offsprings):
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
        number_of_offsprings: int
            number of offsprings made in recombination
        """
        # Asserts
        assert type(population_size) is int, "Population size must be int"

        # Storing
        self.genotype = genotype
        self.population_size = population_size
        self.parent_selection_portion = parent_selection_portion
        self.number_of_offsprings = number_of_offsprings
        self.evaluator = evaluator

        # Creating population
        pop = [] 
        for _ in range(population_size):
            new_genotype = genotype.Genotype(evaluator=evaluator)
            new_genotype.make_permutation(representation)
            pop.append(
                    new_genotype
                )
        self._population = pop 

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
    def parent_selection(self, selection_method="ranked_based_selection"):
        # Init

        # Method selection
        if selection_method == "ranked_based_selection":
            method = self.ranked_based_selection
        
        # Getting children
        # method() # if the store in self is the convention
        parents = method()

        # return/store - Not sure on the convention
        self._parents = parents
        return parents

    # Parent Selection - Methods
    def ranked_based_selection(self):
        # print("__ranked_based_selection__") # - Debug
        scores = self._scores[:]
        N = len(self._population)
        parents = []
        number_of_parents = int(N * self.parent_selection_portion) # making sure to have a whole number
        number_of_parents += int((N * self.parent_selection_portion) % 2) # to make sure to have a even number of parents
        # Making a mapping of the population of indices to the population, this is to be sorted and used for selecting parents
        mapping = [i for i in range(0, N)] # map of population

        # Sorting after scores, rank is the position in the list
        scores, mapping = zip(*sorted(zip(scores, mapping))) # Smallest to largest

        # Selecting
        probabilites = np.array([N - i for i in range(N)]) # Making probabilites for coosing from the invers of traditional ranking
        probabilites = probabilites / np.sum(probabilites) # normalizing 
        selected_map = np.random.choice( # index map of chosen parents
            mapping, p=probabilites, size=number_of_parents, replace=False
            )
        np.random.shuffle(selected_map) # To remove bias in creating couples

        for index in selected_map:
            parents.append(
                self._population[index]
            )

        # print(f"  probabilites: {probabilites}")  # - Debug
        # print(f"  selected_map: {selected_map}")  # - Debug
        self._parents = parents
        return parents
        
    # - Recombination (Crossover)
    def recombination(self, crossover_method="PMX", children_per_couple=2):
        """
        """
        # Init
        parents = self._parents
        offsprings = []
        N = self.number_of_offsprings

        # Asserts
        assert len(parents) % 2 == 0, "(recombination) len of parents must be a even number"
        assert N > len(parents), "(recombination) len of parents must be smaller than the amount of new offsprings"
        if N % (len(parents) * children_per_couple) != 0:
            print("Warning, recombination works best when: N % (len(parents) * children_per_couple)!")

        # Choosing mehtod (only one method implemented!)
        if crossover_method == "PMX":
            method = self.pmx
        else:
            raise NameError("Crossover method does not exist")

        # Applying chosen method
        i = 0
        while len(offsprings) < N:
            couple = (parents[i%len(parents)], parents[(i+1)%len(parents)]) # a pair of parents to create offspring(s)
            for j in range(children_per_couple):
                # With modulo, so that the parents will alternate
                P1 = couple[j%children_per_couple]()
                P2 = couple[(j+1)%children_per_couple]()
                offspring_template = method(P1, P2) # returns a list
                offspring = genotype.Genotype(r=offspring_template) # turns the list to offsping
                offsprings.append(offspring)
            i += 2
    

        # store/return
        # self._offsprings = offsprings
        # return offsprings

        self._population = offsprings # age bias: only offspring

    # Recombination (Crossover) - Methodds
    def pmx(self, P1, P2, c1=None, c2=None):
        # - 1.
        if c1 is None:
            c1 = np.random.randint(0, len(P1))
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

    # - Mutation



    # Crossover
    # evaluation
    # Parent selection
    def print_pop(self):
        print("Population:")
        for obj in self._population:
            print(f"{obj()}")

