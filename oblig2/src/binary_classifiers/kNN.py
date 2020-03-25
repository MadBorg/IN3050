import numpy as np

import IPython

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
    def calc_distances(self, X, method=None):
        if method == "euclidian":
            method = self._euclidian_distance
        else:
            if self.distance_method == "euclidian":
                method = self._euclidian_distance
            else:
                raise NotImplementedError(f"method {method} is not implemented!")

        return method(X)
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
        distances = self.calc_distances(X)
        indices = np.argsort(distances, axis=0)
        targets = self.targets
        classes = np.unique(targets[indices[:k]])
        counts = np.zeros(classes.shape[0]+1)
        # count nearest
        for i in range(k):
            cur_ind = indices[i]
            cur_target = targets[cur_ind]
            try:
                counts[cur_target] += 1
            except IndexError:
                IPython.embed()
        nearest = np.where(counts == np.max(counts))
        # IPython.embed()
        if len(nearest[0]) == 2:
            return self.k_nearest(X, k+1)
        return nearest[0]

    # Choose the majority class for this set
    def classify(self, X, k = None):
        # init
        if not k:
            if self.k:
                k = self.k
            else:
                raise TypeError("K not given!")
        N = X.shape[0] # number of inputs
        nearest = np.zeros(N)
        for i in range(N):
            nearest[i] = self.k_nearest(X[i, :], k)
        return nearest


        


    # predict

