import src.binary_classifiers.binary_classifier as binary_classifier
import numpy as np
import IPython 

class Simple_perceptron(binary_classifier.Binary_Classifier):
    def __init__(self,X, t, learning_rate=0.1, decision_boundary=0.5, epochs=None, diff=0.001):
        self.X = X = self._add_bias(X)
        self.t = t
        (self.k, self.m) = X.shape
        self.W = np.random.rand(self.m)
        self.learning_rate = learning_rate
        self.decision_boundary = decision_boundary
        self.epochs = epochs
        self.diff = diff


    def fit(self, epochs=None, learning_rate=None, diff=None):  # Taining
        # init
        if not epochs:
            if self.epochs:
                epochs = self.epochs
            else:
                raise TypeError("Epocs not given!")
        if not learning_rate:
            if self.learning_rate:
                learning_rate = self.learning_rate
            else:
                raise TypeError("Learning rate not given!")
        if not diff:
            diff = self.diff
        W = self.W
        X = self.X
        t = self.t
        update = np.zeros_like(W)
        learning_rate = self.learning_rate
        diff = self.diff

        # Learn
        # IPython.embed()
        for e in range(epochs):
            y = self.predict(X, W)

            update = learning_rate *  (y - t) @ X
            # update = (learning_rate / self.k) * (y - t) @ X
            if np.sum(np.abs(update)) < diff:
                print(f"Learning stopped by diff, @ epoch {e}!")
                break
            W -= update
        
        # Store / Return
        self.W = W
        return W


    def predict(self, X, W):  # activation function
        return ((X @ W) > self.decision_boundary).astype(int)
        # return (X @ W) > self.decision_boundary # faster, might not be compatible


    def get_accuracy(self, X_test, t_test):
        theta = self.decision_boundary
        N = t_test.shape
        X = self._add_bias(X_test)
        Y = self.predict(X, self.W)
        return (np.sum(Y == t_test) / N)[0]
        

    def _add_bias(self, X):
        return np.insert(X, [2], [-1], axis=1)