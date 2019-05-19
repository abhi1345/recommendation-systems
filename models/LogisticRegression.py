from scipy.special import expit as sigmoid
import numpy as np

class LogisticRegression:
    #Binary Classification
    #Gradient Descent Model
    def __init__(self):
        pass

    def fn(self, x):
        arg = np.array(x).dot(self.w)
        return sigmoid(arg)

    def fit(self, X, y, epsilon=0.001, iterations=5000):
        dim = len(X[0])
        self.w = np.array([0.5]*(dim+1)) #+1 for bias term
        X = np.array(X)
        X = np.insert(X, dim, 1, axis=1)
        for i in range(iterations):
            gradient = np.dot(-X.T, y-self.fn(X))
            self.w -= epsilon*gradient

    def predict(self, X):
        dim = len(X[0])
        X = np.array(X)
        X = np.insert(X, dim, 1, axis=1)
        return np.heaviside(self.fn(X) - 0.5, 1)
