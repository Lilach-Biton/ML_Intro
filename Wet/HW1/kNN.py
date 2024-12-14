from sklearn.base import BaseEstimator,ClassifierMixin
import scipy
import numpy as np

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)
        return self

    def predict(self, X):
        dists = scipy.spatial.distance.cdist(X, self.X_train, 'euclidean')
        idx = np.argpartition(dists, self.n_neighbors+1, axis=1)[:,1:self.n_neighbors +1]           
        y_neighbors = np.copy(self.y_train[idx])
        predictions = None
        # predicted labels (+1 or -1)
        predictions = np.sign(np.mean(y_neighbors, axis=1))
        return predictions