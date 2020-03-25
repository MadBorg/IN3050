import src.binary_classifiers.linear_regression as linear_regression
import src.binary_classifiers.logistic_regression as logistic_regression
import src.binary_classifiers.kNN as kNN
import src.binary_classifiers.simple_perceptron as simple_perceptron
import src.multiclass_classifiers.kNN as kNN_multiclass
import src.data_manager as data

import numpy as np
import matplotlib.pyplot as plt
import IPython


def lin_reg_example():
    print("\n__linear regression__")
    beta = linear_regression.lm(X_train, t2_train)
    outputs = linear_regression.linreg(X_train, t2_train)

    lm = linear_regression.LinReg(X_train, t2_train)
    lm.lm()
    acc_lm = lm.get_accuracy(X_val, t2_val)
    gradient = linear_regression.LinReg(X_train, t2_train)
    gradient.fit(1000, 0.01)
    acc_gradient = gradient.get_accuracy(X_val, t2_val)

    print(f"gradient: acc: {acc_gradient}")
    print(f"lm: acc: {acc_lm}")


def log_reg_example():
    print("\n__Logistic regression__")
    log_reg = logistic_regression.logistic_regression(
        X_train, t2_train,
    )
    log_reg.fit(epochs=1000)
    acc = log_reg.get_accuracy(X_val, t2_val)
    print(f"Log reg: acc: {acc}")


def kNN_example():
    print("\n__kNN__")
    obj = kNN.kNN(X_train, t2_train)
    test = obj.classify(X_val, k=10)
    acc = np.sum((test == t2_val))/t2_val.shape[0]
    print(f"accuracy: {acc}")


def simple_perceptron_example():
    print("\n__simple perceptron__")
    obj = simple_perceptron.Simple_perceptron(X_train, t2_train)
    obj.fit(epochs=1000, learning_rate=0.01, diff=0.001)
    acc = obj.get_accuracy(X_val, t2_val)
    print(f"accuracy: {acc}")


def kNN_multiclass_example(plot=False):
    print("\n__kNN multiclass__")
    obj = kNN_multiclass.kNN(X_train, t_train, k=10)
    acc = np.zeros(400-1)
    for i in range(1, 400):
        acc[i-1] = obj.get_accuracy(X_val, t_val, k=i)
        # print(f"  k={i}, acc: {acc[i-1]}")
    print(f"max acc: {np.max(acc)}, at k: {np.where(acc == np.max(acc))}")
    if plot:
        plt.plot(np.arange(400-1), acc)
        plt.show()
    



if __name__ == "__main__":
    # Init
    X_train = data.X_train
    t_train = data.t_train
    t2_train = data.t2_train
    X_val = data.X_val
    t_val = data.t_val
    t2_val = data.t2_val

    # __Binary__
    lin_reg_example()
    log_reg_example()
    kNN_example()
    simple_perceptron_example()

    # __Multi-class classifiers__
    kNN_multiclass_example(plot=False)

    # plot
    plt.subplot(211)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train, marker=".")
    plt.subplot(212)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, marker=".")
    plt.show()

    # plt.scatter(X_train[:,0], X_train[:,1], s=t2_train)

