import src.binary_classifiers.linear_regression as linear_regression
import src.binary_classifiers.logistic_regression as logistic_regression
import src.binary_classifiers.kNN as kNN
import src.binary_classifiers.simple_perceptron as simple_perceptron
import src.binary_classifiers.binary_classifier as binary_classifier
import src.multiclass_classifiers.kNN as kNN_multiclass
import src.multiclass_classifiers.log_reg_one_vs_rest as log_reg_one_vs_rest
import src.binary_classifiers.multilayer_neural_networks as multilayer_neural_networks
import src.data_manager as data

import numpy as np
import scipy.stats as stats
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

def log_reg_one_vs_rest_example():
    print("__Log reg one vs rest__")
    obj = log_reg_one_vs_rest.Multinomial_Logistic_Regression(
        X_train, t_train
    )
    obj.fit(epochs=300)
    acc = obj.get_accuracy(X_val, t_val)
    print(f"acc: {acc}")
    
def non_linear_features():
    print("__non_linear_features__")
    # X1_squared = binary_classifier.add_feature(
    #     X_train, X_train[:,0] **2
    #     )
    # X2_squared = binary_classifier.add_feature(
    #     X_train, X_train[:, 1]**2
    # )
    # X1_X2 = binary_classifier.add_feature(
    #     X_train,  X_train[:, 0] * X_train[:, 1]
    # )
    X1_squared = X_train[:,0] **2
    X2_squared = X_train[:, 1]**2
    X_X =  X_train[:, 0] * X_train[:, 1]
    X_non_lin = binary_classifier.add_features(X_train, [X1_squared, X2_squared, X_X])

    X_val1_squared = X_val[:,0] **2
    X_val2_squared = X_val[:, 1]**2
    X_val_X_val =  X_val[:, 0] * X_val[:, 1]
    X_val_non_lin = binary_classifier.add_features(X_val, [X_val1_squared,X_val2_squared, X_val_X_val])

    # Might be better in loop, but probably faster to implement without
    # Non lin
    kNN_non_lin = kNN.kNN(X_non_lin,t2_train, k=17)
    lin_reg_non_lin = linear_regression.LinReg(X_non_lin, t2_train)
    log_reg_non_lin = logistic_regression.logistic_regression(X_non_lin, t_train)
    simpl_percept_non_lin = simple_perceptron.Simple_perceptron(X_non_lin, t_train)

    lin_reg_non_lin.fit(epochs=500, gamma=0.1)
    log_reg_non_lin.fit(epochs=500)
    simpl_percept_non_lin.fit(epochs=500)

    acc_kNN_non_lin = kNN_non_lin.get_accuracy(X_val_non_lin, t2_val)
    acc_lin_reg_non_lin = lin_reg_non_lin.get_accuracy(X_val_non_lin, t2_val)
    acc_log_reg_non_lin = log_reg_non_lin.get_accuracy(X_val_non_lin, t2_val)
    acc_simpl_percept_non_lin = simpl_percept_non_lin.get_accuracy(X_val_non_lin, t2_val)

    # Controll
    kNN_controll = kNN.kNN(X_train,t2_train, k=17)
    lin_reg_controll = linear_regression.LinReg(X_train, t2_train,)
    log_reg_controll = logistic_regression.logistic_regression(X_train, t_train)
    simpl_percept_controll = simple_perceptron.Simple_perceptron(X_train, t_train)

    lin_reg_controll.fit(epochs=500,  gamma=0.1)
    log_reg_controll.fit(epochs=500)
    simpl_percept_controll.fit(epochs=500) 

    acc_kNN_controll = kNN_controll.get_accuracy(X_val, t2_val)
    acc_lin_reg_controll = lin_reg_controll.get_accuracy(X_val, t2_val)
    acc_log_reg_controll = log_reg_controll.get_accuracy(X_val, t2_val)
    acc_simpl_percept_controll = simpl_percept_controll.get_accuracy(X_val, t2_val)

    # Making table
    print("Table over accuracies for linear and non-linear")
    print("algorithm     | Linear | Non-linear")
    print(f"kNN           | {acc_kNN_controll}   | {acc_kNN_non_lin}")
    print(f"lin_reg       | {acc_lin_reg_controll}   | {acc_lin_reg_non_lin}")
    print(f"log_reg       | {acc_log_reg_controll}   | {acc_log_reg_non_lin}")
    print(f"simpl_percept | {acc_simpl_percept_controll}   | {acc_simpl_percept_non_lin}")


    # kNN_X1_squared = kNN.kNN(X1_squared,t2_train, k=17)
    # lin_reg_X1_squared = linear_regression.LinReg(X1_squared,t2_train, gamma=0.1)
    # log_reg_X1_squared = logistic_regression.logistic_regression(X1_squared, t2_train)
    # simpl_percept_X1_squared = simple_perceptron.Simple_perceptron(X1_squared, t2_train)

    # kNN_X2_squared = kNN.kNN(X2_squared ,t2_train, k=17)
    # lin_reg_X2_squared = linear_regression.LinReg(X2_squared, t2_train, gamma=0.1)
    # log_reg_X2_squared = logistic_regression.logistic_regression(X2_squared, t2_train)
    # simpl_percept_X2_squared = simple_perceptron.Simple_perceptron(X2_squared, t2_train)

    # kNN_X1_X2 = kNN.kNN(X1_X2, t2_train, k=17)
    # lin_reg_X1_X2 = linear_regression.linreg(X1_X2, t2_train, gamma=0.1)
    # log_reg_X1_X2 = logistic_regression.logistic_regression(X1_X2, t2_train)
    # simpl_percept_X1_X2 = simple_perceptron.Simple_perceptron(X1_X2, t2_train)

class Scaler:
    def __init__(self, X):
        """
        Storing scaling data so that we can scale datasets
        """
        (k, m) = X.shape
        self.mean = (1/k) * np.sum(X, axis=0)
        self.sd = np.std(X, axis=0)

    def scale(self, X):
        return (X - self.mean) / self.sd

def logit_activation(X, W, binary=False, decision_boundary=0.5):
    alfa = 1 / (1+ np.exp(-(X @ W))) # 1/(1+np.exp(-(beta * (X@W))))
    # IPython.embed(header="logit")
    if binary:
        return np.argmax(alfa, axis=1)
    else:
        return alfa

def one_round_of_training(X, target, dim_hidden=6, eta=0.01):
    """
    eta: float
        learningrate
    """
    # init
    if not dim_hidden:
        dim_hidden = int(input("Dim hidden: "))
    (P, dim_in) = X.shape # P: number of training instances, dim_in: number of features (L in marshland)
    try:
        (P, dim_out) = target.shape
    except ValueError:
        P = target.shape[0]
        dim_out = 2
    scale = Scaler(X)
    X_scaled = scale.scale(X)
    X_train = data.add_bias(X_scaled)
    t = target[:, None]

    # Making the weights
    # L1_weight_range = (-1/np.sqrt(dim_in),1/np.sqrt(dim_in))
    # L2_weight_range = (-1/np.sqrt(dim_in),1/np.sqrt(dim_in))

    L1_weight_range = (-1, 1)
    L2_weight_range = (-1, 1)

    L1_W = (L1_weight_range[1] - L1_weight_range [0]) * np.random.random(size=(dim_in+1, dim_hidden)) *  L1_weight_range[0]
    L2_W = (L2_weight_range[1] - L2_weight_range [0]) * np.random.random(size=(dim_hidden, dim_out)) *  L2_weight_range[0]

    alfa = hidden_activations = logit_activation(X_train, L1_W)
    Y = output_activations = logit_activation(hidden_activations, L2_W)

    # IPython.embed()

    delta_o = (t - Y) * Y * (1-Y)
    delta_h = alfa*(1-alfa)* (delta_o @ L2_W.T)

    update_L1 = eta * X_train.T @ delta_h
    update_L2 = eta * alfa.T @ delta_o

    new_L1_W = L1_W + update_L1
    new_L2_W = L2_W + update_L2

    print(f"Weights:")
    print(f"  L1:\n{L1_W}\n  L2:\n{L2_W}\nNew Weights:\n  new L1:\n{new_L1_W}\n  new L2:\n{new_L2_W}")
    print(f"  Update 1:\n{update_L1}\n  Update 2:\n{update_L2}")

def MNN_example(X_train, t_train, X_test, t_test):
    obj = multilayer_neural_networks.MNNClassifier()
    obj.fit(X_train, t_train)
    acc = obj.accuracy(X_test, t_test)
    IPython.embed("MNN_example")

if __name__ == "__main__":
    # Init
    X_train = data.X_train
    t_train = data.t_train
    t2_train = data.t2_train
    X_val = data.X_val
    t_val = data.t_val
    t2_val = data.t2_val

    # # __Binary__
    # lin_reg_example()
    # log_reg_example()
    # kNN_example()
    # simple_perceptron_example()

    # __Multi-class classifiers__
    # kNN_multiclass_example(plot=False)
    # log_reg_one_vs_rest_example()

    # __Adding non linear-features
    # non_linear_features()

    # Part 2
    scale = Scaler(X_train)
    X_train_scaled = scale.scale(X_train)

    # one_round_of_training(X_train_scaled, t2_train)
    MNN_example(X_train, t_train, X_val, t_val)



