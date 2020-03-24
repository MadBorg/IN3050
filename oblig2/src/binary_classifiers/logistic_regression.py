import numpy as np

class logistic_regression:
    def __init__(self,X, t, learning_rate=0.1, decision_boundary=0.5, epochs=None):
        self.X = self._add_bias(X)
        self.t = t # targets
        learning_rate = 0.1
        self.decision_boundary = decision_boundary

        # Optional variables
        if epochs:
            self.epochs

    # fit: learn
    def fit(self, y):
        # Init
        X = self.X
        t = self.t
        (k, m) = X.shape
        self.W = np.zeros(m)
        update = np.zeros_like(W)

        # learning
        for e in range(epochs):
            # TODO : implement loop for updating weights
    # loss function
    """
    We need a loss function that expresses, for an observation x, how close the classifier output (y_hat = sigma(w @ x + b)) is to the correct output (y, which is 0 or 1). 
    """
    def L(self, y_hat, y, method="cross-entropy"): # loss function
        """
        Loss function:
            How much y_hat differs from the true y
        """
        # Choosing method (if more where to be added)
        if method == "cross-entropy":
            method = self.cross_entropy
    # - Cross entropy loss function'
    def cross_entropy(self, y_hat, y):
        """
        cross-entropy loss function, using gradient decent
        """

    def _add_bias(self, X):
        return np.insert(X, [2], [-1], axis=1)
        

    # Predict
    # y = sigma(z) = 1/(1+exp(-z))

    # Z = w @ x + b, b: bias