import random

class Genotype:
    """
    Representation:
        Permutation
    Evaluator:
        I have chosen to generalize so evaluator is not predefined in the class but must be given, in my task that is in the main.
    """
    def __init__(self, representation, evaluator=None):
        r = representation[:] # making sure not to change the original list
        random.shuffle(r)
        self._r = tuple(r)
        if evaluator:
            self.evaluator = evaluator
        
        self._score = None

    def __call__(self):
        return self.r

    # mutation - method
    # Swap Mutation
    # Insert Mutation
    # Scramble Mutation
    # Inversion Mutation

    # Score 
    def calculate_score(self, df):
        self._score = self.evaluator(self.r, df)     
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