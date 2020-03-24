import numpy as np

import IPython


class LinReg:
    def __init__(self, inputs, targets, X_val=None, t_val=None, gamma=None, diff=0.001):
        self.X = self._add_bias(inputs)
        self.t = targets
        self.W = None
        self.diff = diff
        if X_val:
            self.X_val = X_val
        if t_val:
            self.t_val = t_val
        if gamma:
            self.gamma = gamma

    def __call__(self, X):
        return self.predict(X)

    def fit(self, epochs, gamma=None):     # Gradient decent
        # init
        if gamma:
            self.gamma = gamma
        else:
            if self.gamma:
                gamma = self.gamma
            else:
                raise TypeError(
                    "fit missing required argument gamma, must be given in init or in fit")

        T = self.t
        diff = self.diff
        X = self.X
        (k, m) = X.shape
        self.W = np.zeros(m)
        update = np.zeros_like(self.W)

        # Fiting the model
        for i in range(epochs):
            update = (gamma / k) * (X.T @ (self(X) - T)) # self(): predict

            sum_update = np.sum(np.abs(update))
            # print(f"Epoch: {i}:\n  W: {self.W}\n  sum(update): {sum_update}\n  diff: {diff}")
            if sum_update < diff:
                print(f"Fit break hit after {i} epochs")
                break
            self.W -= update

        print(f"Epochs run: {i}")
        # Store / Return
        return self.W

    def lm(self):
        """
        Normal linear regression
        """
        inputs = self.X
        targets = self.t

        self.W = beta = np.dot(np.dot(np.linalg.inv(
            np.dot(np.transpose(inputs), inputs)), np.transpose(inputs)), targets)

    def get_mse(self):
        # init
        t = self.t
        N = t.shape[0]
        X = self.X
        # --

        mse = (1/N) * np.sum((t - np.sum(self(X)))**2)  # self(): predict
        return mse

    def predict(self, X):
        return X @ self.W

    def get_accuracy(self, X_test, t_test):
        theta = 0.5
        N = t_test.shape
        X = self._add_bias(X_test)
        Y = self(X)  # Predict
        Y_binary = Y > theta
        # IPython.embed()
        return (np.sum((Y_binary == t_test).astype(int)) / N)[0]

    def _add_bias(self, X):
        return np.insert(X, [2], [-1], axis=1)


def linreg(inputs, targets):
    inputs = np.concatenate(
        (inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
    beta = np.dot(np.dot(np.linalg.inv(
        np.dot(np.transpose(inputs), inputs)), np.transpose(inputs)), targets)
    outputs = np.dot(inputs, beta)
    return outputs


def lm(inputs, targets):
    inputs = np.concatenate(
        (inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
    beta = np.dot(np.dot(np.linalg.inv(
        np.dot(np.transpose(inputs), inputs)), np.transpose(inputs)), targets)
    return beta
