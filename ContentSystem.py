import numpy as np
import pandas as pd
from models.kNN import *

class ContentSystem:

    def __init__(self):
        self.trained = False
        pass

    def __repr__(self):
        return self.name + "Trained: {}".format(self.trained)

    def train(self):
        self.trained = True

    def 
