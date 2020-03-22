import src.binary_classifiers.linear_regression as linear_regression
import src.data_manager as data

import IPython

def lin_reg():
    # __Binary Classifiers__
    # Linear regression
    beta = linear_regression.lm(X_train, t2_train)
    outputs = linear_regression.linreg(X_train, t2_train)

    lm = linear_regression.LinReg(X_train, t2_train)
    lm.lm()
    acc_lm = lm.get_accuracy(X_val, t_val)
    gradient = linear_regression.LinReg(X_train, t2_train)
    gradient.fit(10000, 0.05)
    acc_gradient = gradient.get_accuracy(X_val, t_val)
    
    mse_gradient = gradient.get_mse()
    mse_lm = lm.get_mse()

    print(f"gradient: acc: {acc_gradient}, mse: {mse_gradient}")
    print(f"lm: acc: {acc_lm}, mse: {mse_lm}")

if __name__ == "__main__":
    # Init
    X_train = data.X_train
    t_train = data.t_train
    t2_train = data.t2_train
    X_val = data.X_val
    t_val = data.t_val

    # Linear regression
    lin_reg()
    # kNN
    
