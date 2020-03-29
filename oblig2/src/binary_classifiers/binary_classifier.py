import numpy as np
import IPython

class Binary_Classifier:
    def __init__(self, training_set, targets, learningrate=None, diff=0.001, decision_boundary=0.5, bias=False):
        self.X = training_set
        self.t = self.targets
        self.learningrate = learningrate
        self.diff = diff
        self.decision_boundary

    def _add_bias(self, X):
        return np.append(X, np.zeros((X.shape[0],1))-1, axis=1)

    def get_accuracy(self, X_val, t_val, k=None):
        Y = self.classify(X_val, k)
        return np.sum((Y == t_val))/t_val.shape[0]

def add_feature(X, feature): # could be a static function, but would rather fit in a class for training sets
    try:
        return np.append(X, feature[:, None], axis=1)
    except ValueError:
        IPython.embed()
        exit()

def add_features(X, features):
    """
    features: list<numpy_array>
        list of features
    """
    for feature in features:
        X = add_feature(X, feature)
    return X

if __name__ == "__main__":
    X = np.zeros((5,2))
    f = np.zeros((5, 1)) -1
    IPython.embed()
    print(f"X: {X}\nf: {f}\nadded f: {add_feature(X, f)}")
