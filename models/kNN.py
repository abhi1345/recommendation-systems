#K Nearest Neighbors Classifier

class kNearestNeighbors:
    def __init__(self):
        pass

    def fit(self, X):
        self.X = X

    def distance(self, u, v):
        #Euclidean
        s = 0
        for i in range(len(u)):
            s += (u[i] - v[i])**2
        return s**(0.5)

    def classify(self, z, n=1):
        heap = []
        heapmax = float("inf")
        for i in range(len(self.X)):
            x = self.X[i]
            d = self.distance(x, z)
            if heap:
                hmax = max(heap, key=lambda x: x[0])
                hmax_d = hmax[0]
            if len(heap) < n:
                heap.append((d, i))
            elif d < hmax_d:
                heap.remove(hmax)
                heap.append((d, i))
        return heap
