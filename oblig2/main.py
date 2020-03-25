import src.binary_classifiers.linear_regression as linear_regression
import src.binary_classifiers.logistic_regression as logistic_regression
import src.binary_classifiers.kNN as kNN
import src.binary_classifiers.simple_perceptron as simple_perceptron
import src.data_manager as data

import numpy as np
import IPython

def lin_reg_example():
    print("__linear regression__")
    beta = linear_regression.lm(X_train, t2_train)
    outputs = linear_regression.linreg(X_train, t2_train)

    lm = linear_regression.LinReg(X_train, t2_train)
    lm.lm()
    acc_lm = lm.get_accuracy(X_val, t2_val)
    gradient = linear_regression.LinReg(X_train, t2_train)
    gradient.fit(1000, 0.01)
    acc_gradient = gradient.get_accuracy(X_val, t2_val)
    
    mse_gradient = gradient.get_mse()
    mse_lm = lm.get_mse()

    print(f"gradient: acc: {acc_gradient}, mse: {mse_gradient}")
    print(f"lm: acc: {acc_lm}, mse: {mse_lm}")

def log_reg_example():
    print("__Logistic regression__")
    log_reg = logistic_regression.logistic_regression(
        X_train, t2_train,
    )
    log_reg.fit(epochs=1000)
    acc = log_reg.get_accuracy(X_val, t2_val)
    print(f"Log reg: acc: {acc}")

def kNN_example():
    print("__kNN__")
    obj = kNN.kNN(X_train, t2_train)
    test = obj.classify(X_val, k=10)
    acc = np.sum((test == t2_val))/t2_val.shape[0]
    print(f"accuracy: {acc}")

def simple_perceptron_example():
    print("__simple perceptron__")
    obj = simple_perceptron.Simple_perceptron(X_train, t2_train)
    obj.fit(epochs=1000, learning_rate=0.01, diff=0.001)
    acc = obj.get_accuracy(X_val, t2_val)
    print(f"accuracy: {acc}")

if __name__ == "__main__":
    # Init
    X_train = data.X_train
    t_train = data.t_train
    t2_train = data.t2_train
    X_val = data.X_val
    t_val = data.t_val
    t2_val = data.t2_val

    # Linear regression
    lin_reg_example()
    # Logistic regression
    log_reg_example()
    # kNN
    kNN_example()
    # simple_perceptron_example
    simple_perceptron_example()

