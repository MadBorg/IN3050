import src.binary_classifiers.binary_classifier as binary_classifier

import numpy as np

import IPython

class logistic_regression(binary_classifier.Binary_Classifier):
    def __init__(self,X, t, learning_rate=0.1, decision_boundary=0.5, epochs=None, diff=0.001):
        self.X = self._add_bias(X)
        self.t = t # targets
        self.learning_rate = learning_rate
        self.decision_boundary = decision_boundary
        self.epochs = epochs
        self.diff = diff

    # fit: learn
    def fit(self, epochs=None, learning_rate=None, diff=None):
        # Init
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
        X = self.X
        t = self.t
        (k, m) = X.shape
        W = np.zeros(m)
        update = np.zeros_like(W)

        # learning
        for e in range(epochs):
            update = (learning_rate / k) * (X.T @ (self.predict(X, W) - self.t))
            # update = (learning_rate / 1) * (X.T @ (self.predict(X, W)- self.t))

            
            sum_update = np.sum(np.abs(update))
            if sum_update < diff:
                print(f"Fit break hit after {e} epochs")
                break
            W -= update
        
        self.W = W
        return W

    def predict(self, X, W): # log
        try:
            return 1/(1+np.exp(-(X @ W)))
        except ValueError:
            print("log reg predict - value error!")
            IPython.embed()
            exit()
    def get_accuracy(self, X_test, t_test):
        theta = self.decision_boundary
        N = t_test.shape
        X = self._add_bias(X_test)
        Y = self.predict(X, self.W)
        Y_binary = Y > theta
        return (np.sum(Y_binary == t_test) / N)[0]
        
    # def _add_bias(self, X):
    #     return np.insert(X, [2], [-1], axis=1)


    # # loss function
    # """
    # We need a loss function that expresses, for an observation x, how close the classifier output (y_hat = sigma(w @ x + b)) is to the correct output (y, which is 0 or 1). 
    # """
    # def L(self, y_hat, y, method="cross-entropy"): # loss function
    #     """
    #     Loss function:
    #         How much y_hat differs from the true y
    #     """
    #     # Choosing method (if more where to be added)
    #     if method == "cross-entropy":
    #         method = self.cross_entropy
    # # - Cross entropy loss function'
    # def cross_entropy(self, y_hat, y):
    #     """
    #     cross-entropy loss function, using gradient decent
    #     """


        

    # Predict
    # y = sigma(z) = 1/(1+exp(-z))

    # Z = w @ x + b, b: bias