import src.multiclass_classifiers.multiclass_classifier as multiclass

import numpy as np
import IPython

class kNN(multiclass.Multiclass_Classifier):
    def __init__(self, X, t, k=None, distance_method = "euclidian"):
        super().__init__(X, t, bias=False)
        self.k = k
        self.distance_method = distance_method
    
    def predict(self, values, k=None):
        # init
        if not k:
            if self.k:
                k = self.k
            else:
                raise  TypeError("K not given!")
        data = self.X
        if len(values.shape) == 2:
            N = values.shape[0]
        else:
            return self._predict_single(values, k)
        closest = np.ones(N)*-1 # setting default

        #asserts
        assert self.X.shape[1] == values.shape[1]

        # kNN
        for i in range(N):
            # compute distances
            # Identify the nearest neighbors
            distances = self.get_distances(values[i])
            indices = np.argsort(distances, axis=0)
            classes = self.get_classes(self.t[indices[0:k]])

            if len(classes) == 1:
                closest[i] = classes[0]  # np.unique(classes) 
            else:
                counts = np.zeros(max(classes)+1)  # works when the map for classes is integers, can use a hash like structure if not. Might also make outside of loop
                for j in range(k):
                    counts[self.t[indices[j]]] += 1
                # closest[i] = np.max(counts) # not the index
                Y = np.where(counts == np.max(counts)) # the index of the NN class
                # IPython.embed()
                if len(Y[0]) == 1:
                    closest[i] = Y[0]
                else:
                    closest[i] = self.predict(values[i, :], k=k+1)[0]    
        return closest
        
    def _predict_single(self, value, k):
        distances = self.get_distances(value)
        indices = np.argsort(distances)
        classes = self.get_classes(self.t[indices[0:k]])
        if len(classes) == 1:
            return classes[0]
        else:
            counts = np.zeros(max(classes)+1)
            for j in range(k):
                counts[self.t[indices[j]]] += 1
            Y = np.where(counts == np.max(counts))
            if len(Y[0]) == 1:
                return Y[0]
            else:
                return self._predict_single(value, k+1)

        


    def get_distances(self, inputs, data=None):
        # init
        if self.distance_method == "euclidian":
            method = self._euclidian_distance
        else:
            raise TypeError("distance method not given!")
        if not data:
            data = self.X
        return method(inputs, data)
    
    def _euclidian_distance(self, inputs, data):
        return np.sum( (data - inputs)**2, axis=1 )

    def classify(self, values, k=None):
        return self.predict(values, k=k)



