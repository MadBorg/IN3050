import numpy as np

import IPython

class MNNClassifier:
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self,eta = 0.001, dim_hidden = 6):
        """
        Intialize the hyperparameters

        dim_hidden: int
            Might be good to change to a iterable of ints if we want different dim on different layers 
        """
        self.eta = eta
        self.dim_hidden = dim_hidden
        self.layers = 2
        # Should you put additional code here?
        # self.num_hidden_layers = num_hidden_layers

    def fit(self, X_train, t_train, epochs = 100):
        """Intialize the weights. Train *epochs* many epochs."""
        
        # Initilaization
        # Fill in code for initalization
        # IPython.embed()
        self.X_train = X_train
        self.t_train = t_train[:, None]
        classes = np.unique(t_train)
        dim_out = classes.shape[0]
        (P, dim_in) = X_train.shape 

        #Scaling X
        # self.scaler = Scaler(X_train)
        # self.X = X = self.scaler.scale(X_train)
        # init Weights, Since the weights can have different dimensions, i am making a list of weight matrixes
        self.X = X = X_train
        weights = []
        nodes_pr_layer = [dim_in, self.dim_hidden, dim_out]


        for i in range(len(nodes_pr_layer)-1):
            # W_scale = 1/nodes_pr_layer[i]
            # weights.append(
            #     (W_scale - -W_scale) * np.random.random(size=(nodes_pr_layer[i]+1, nodes_pr_layer[i+1])) * -W_scale
            # )
            weights.append(
                np.random.random(size=(nodes_pr_layer[i]+1, nodes_pr_layer[i+1]))
            )
        self.weights = weights
        # IPython.embed(header="weights")

        # learning
        for e in range(epochs):
            # Run one epoch of forward-backward
            #Fill in the code
            activations = self.forward(X)
            self.backward(X, activations)

            # pass

    def forward(self, X):
        """Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        #Fill in the code
        activations = [X]
        for l in range(self.layers):
            # cur_X = self._add_bias(self._logit_activation(cur_X, self.weights[l]))
            # IPython.embed(header=f"forward l:{l}")
            activations[l] = self._add_bias(activations[l])
            activations.append(
                 self._logit_activation(activations[l], self.weights[l])
            )
        return activations

    def backward(self, X, activations):
        """
        updating the weights
        """
        X = activations[0]
        alfa = activations[1]
        Y = activations[2]
        t = self.t_train

        
        delta_o = (t - Y) * Y * (1 - Y)
        # delta_h = activations[0] * (1- activations[0]) * delta_o.T @ self.weights[0]
        delta_h = alfa * (1- alfa) * (delta_o @ self.weights[1].T)

         
        update_w1 = self.eta * X.T @ delta_h[:,:-1]
        update_w2= self.eta * alfa.T @ delta_o

        # IPython.embed(header="bacward")
        self.weights[0] += update_w1
        self.weights[1] += update_w2

        # - for multiple layers not working
        # update_o = self.eta * X.T @ delta_o
        # IPython.embed(header="backward")
        # self.weights[-1] += update_o
        # for l in range(self.num_hidden_layers):
        #     alfa = activations[l]
        #     delta_h =  alfa*(1-alfa)* (delta_o @ self.weights[l].T)
        #     update_h = self.eta * alfa.T @ delta_h
        #     updates.append(update_h)
        #     self.weights[l] += update_h
        # updates.append(update_o)

        # return updates

    def classify(self, X):
        r = self.forward(X)[-1]
        # IPython.embed(header="classify")
        return np.argmax(r, axis=1)

    def accuracy(self, X_test, t_test):
        """Calculate the accuracy of the classifier on the pair (X_test, t_test)
        Return the accuracy"""
        #Fill in the code
        c = self.classify(X_test)
        # IPython.embed(header="acc")
        return np.sum(c == t_test) / t_test.shape[0]

    def _logit_activation(self, X, W):
        return 1 / (1 + np.exp(-1 * (X @ W))) # 1/(1+np.exp(-(beta * (X@W))))


    def _add_bias(self, X):
        return np.append(X, np.zeros((X.shape[0],1))-1, axis=1)

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