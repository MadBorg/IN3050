import random
# import numpy as np 

class Genotype():
    """
    Representation:
        Permutation
    Evaluator:
        I have chosen to generalize so evaluator is not predefined in the class but must be given, in my task that is in the main.
    """
    def __init__(self,r=None, evaluator=None):
        if r:
            self._r = r
        else:
            self._r = None
        if evaluator:
            self.evaluator = evaluator
        
        self._score = None

    def __call__(self):
        return self.r

    def __len__(self):
        return len(self())
    
    def make_permutation(self, representation):
        """
        Making random premutation of the representation
        """
        r = representation[:]
        random.shuffle(r)
        self._r = tuple(r)

    # mutation - method

    # Swap Mutation
    def swap_mutation(self):
        # Init
        locus1 = locus2 = 0
        index_options = [i for i in range(0,len(self))]

        # Coosing random indices
        locus1, locus2 = random.sample(index_options, 2) # Choosing two

        # Swaping
        tmp = self.r[locus1]
        self.r[locus1] = self.r[locus2]
        self.r[locus2] = tmp
        
        
    # Insert Mutation
    # Scramble Mutation
    # Inversion Mutation

    # Score 
    def calculate_score(self, df):
        score = self.evaluator(self.r, df)     
        self._score = score
        return score   

    @property
    def evaluator(self):
        return self._fit
    @evaluator.setter
    def evaluator(self, fun):
        self._fit = fun

    @property
    def r(self):
        return self._r

    @property
    def score(self):
        if self._score:
            return self._score
        else:
            raise ValueError("Score not calculated!")