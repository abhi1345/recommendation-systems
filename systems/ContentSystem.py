import numpy as np
import pandas as pd
from models.kNN import *

class ContentSystem:

    def __init__(self):
        self.trained = False
        self.name = "Model Based Recommendation System."
        pass

    def __repr__(self):
        return self.name + " Trained: {}".format(self.trained)

    def train(self):
        self.dataset = pd.read_csv("data/mtcars.csv")
        self.dataset.columns = ["name", "mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"]
        X = self.dataset.ix[:, (1,3,4,6)].values
        self.model = kNearestNeighbors()
        self.model.fit(X)
        self.trained = True

    def generate_recommendations(self, x, n=5):
        if not self.trained:
            self.train()
        neighbors = self.model.classify(x, n)
        n_index = [x[1] for x in neighbors]
        return self.dataset.ix[n_index, :]


test = [15, 300, 160, 3.2]
sys = ContentSystem()
print(sys.generate_recommendations(test))
