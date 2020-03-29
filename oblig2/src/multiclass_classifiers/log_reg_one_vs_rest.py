import src.multiclass_classifiers.multiclass_classifier as multiclass
import src.binary_classifiers.logistic_regression as logistic_regression

import numpy as np
import IPython


class Multinomial_Logistic_Regression(multiclass.Multiclass_Classifier):
    def __init__(self, X, t, learning_rate=0.1, decision_boundary=0.5, epochs=None, diff=0.001):
        self.X = X
        self.t = t
        self.classes= np.unique(t)
        self.learning_rate = learning_rate
        self.decision_boundary = decision_boundary
        self.epochs = epochs
        self.diff = diff

        self.N_classes = self.classes.shape[0]

        # one log reg per class
        self.log_reg_objs = []
        for i in range(self.N_classes):
            tmp_obj = logistic_regression.logistic_regression(
                self.X, self.t == i, learning_rate, decision_boundary=decision_boundary, epochs=epochs, diff=diff
            )
            self.log_reg_objs.append(tmp_obj)
        IPython.embed(header="init multi log reg")
        
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
        
        for obj in self.log_reg_objs:
            obj.fit(epochs, learning_rate, diff)

    def classify(self, values):
        (k, m) = np.shape(values)
        preds = np.zeros((self.N_classes, k))
        classified = np.zeros(k)
        
        for i in range(self.N_classes):
            p = (self.log_reg_objs[i].predict(self._add_bias(values), self.log_reg_objs[i].W))
            preds[i, :] = p
        classified = np.argmax(preds, axis=0)
        return classified

    def get_accuracy(self, X_val, t_val):
        theta = self.decision_boundary
        N = t_val.shape
        X = X_val
        Y = self.classify(X)
        return np.sum(Y == t_val) / N



        
        




