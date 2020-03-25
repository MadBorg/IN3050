import numpy as np

class Multiclass_Classifier:
    """
    Parent class
    """
    
    def __init__(self, X, t, bias=True):
        if bias:
            self.X = self._add_bias(X)
        else:
            self.X = X
        self.t = t
        (self.k, self.m) = self.X.shape

    def _add_bias(self, X):
        return np.insert(X, [2], [-1], axis=1)

    def get_classes(self, X=None):
        if X is None:
            return np.unique(self.t)
        else:
            return np.unique(X)

    def classify(self, values):
        raise NotImplementedError("Classify is not implemented")

    def get_accuracy(self, X_val, t_val, **kwargs):
        Y = self.classify(X_val, **kwargs)
        return np.sum((Y == t_val).astype(int)) / t_val.shape[0]