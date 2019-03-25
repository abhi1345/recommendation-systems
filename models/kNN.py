class kNearestNeighbors:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def distance(self, u, v):
        #Euclidean
        s = 0
        for i in range(len(u)):
            s += (u[i] - v[i])**2
        return s**(0.5)
    
    def classify(self, z):
        min_dist = float('inf')
        answer = 0
        for i in range(len(self.X)):
            x = self.X[i]
            yhat = self.y[i]
            d = self.distance(x, z)
            if d < min_dist:
                min_dist = d
                answer = yhat
        return answer
