import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
import random
import matplotlib.pylab as plt
import IPython

from sklearn.datasets import make_blobs

# Making data for classification
# X: Samples, t: class per sample
X, t = make_blobs(n_samples=[400,800,400], centers=[[0,0],[1,2],[2,3]], 
                  n_features=2, random_state=2019)

indices = np.arange(X.shape[0])
random.seed(2020)
random.shuffle(indices)
indices[:10]

def add_bias(X):
    return np.append(X, np.zeros((X.shape[0],1))-1, axis=1)

# split values to training, validation, final testing
X_train = X[indices[:800],:]
X_val = X[indices[800:1200],:]
X_test = X[indices[1200:],:]
t_train = t[indices[:800]]
t_val = t[indices[800:1200]]
t_test = t[indices[1200:]]

# Making a second dataset by merging the two smaller classes, this will be a binary set
t2_train = t_train == 1
t2_train = t2_train.astype('int')
t2_val = (t_val == 1).astype('int')
t2_test = (t_test == 1).astype('int')

if __name__ == "__main__":
    # IPython.embed()
    # Plot the two training sets.
    
    fig = plt.figure(figsize=(2,1))
    ax1 = fig.add_subplot(211, title = "X_train, t_train")
    ax1.scatter(X_train[:,0], X_train[:,1], marker='.', c=t_train)

    ax2 = fig.add_subplot(212, title = "X_train, t2_train")
    ax2.scatter(X_train[:,0], X_train[:,1], marker='.', c=t2_train) 

    plt.show()
    
    # IPython.embed()