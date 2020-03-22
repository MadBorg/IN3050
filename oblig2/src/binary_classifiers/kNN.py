import numpy as np

class kNN:
    distance_methods = ["euclidian"]
    def __init__(self, data, targets, k=None, distance_method = "euclidian"):
        self.data = data
        self.targets = targets
        if k:
            self.k = k
        if not distance_method in self.distance_methods:
            raise NotImplementedError(f"Distance method {distance_method} not implemented")
        self.distance_method = distance_method

    # distances
    def calc_distances(self, X, method=""):
        if self.distance_method == "euclidian":
            method = self._euclidian_distance(X)
        else:
            raise NotImplementedError(f"method {method} is not implemented!")

    # - Euclidian distance  (pythagoras)
    def _euclidian_distance(self, X):
        return np.sum( (self.data - X)**2, axis=1 )

    # pick the k nearest
    def k_nearest(self, X, k=None):
        # init
        if not k:
            if self.k:
                k = self.k
            else:
                raise TypeError("K not given!")
        counter = {*self.classes}
        distances = self.calc_distances(X)
        indices = np.argsort(distances, axis=0)

        # count nearest
        # TODO : targets
        for ind in indices[:k]:






    # Choose the majority class for this set
    def classify(self):
        


    # predict

