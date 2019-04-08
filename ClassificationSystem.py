import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from models.LogisticRegression import *

global_var_user = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

#Classification-based Recommendation system
#Recommends user for certain product based on user's features.
#Uses logistic regression.
class ClassificationSystem:
    def __init__(self):
        bank_full = pd.read_csv('data/bank_full_w_dummy_vars.csv')
        X = bank_full.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
        y = bank_full.ix[:,17].values
        LogReg = LogisticRegression()
        LogReg.fit(X, y)
        self.model = LogReg

    def __repr__(self):
        return "Classification Based Product Recommendation System. Uses Logistic Regression to determine user-product match."

    def generate_recommendations(self, new_user=global_var_user):
        y_pred = self.model.predict(np.array([new_user]))
        map = ["no", "yes"]
        return map[int(y_pred)]
