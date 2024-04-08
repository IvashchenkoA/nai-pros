import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k, v_train, cl_train):
        self.k = k
        self.v_train = v_train
        self.cl_train = cl_train

    def evaluate(self, v_test):
        results = []
        for v in v_test:
            eucdDistances = [np.linalg.norm(v - v_train) for v_train in self.v_train]
            nearestNeighbors = np.argsort(eucdDistances)[:self.k]
            nearestCLabels = [self.cl_train[i] for i in nearestNeighbors]
            most_common_label = Counter(nearestCLabels).most_common(1)[0][0]
            results.append(most_common_label)
        return results

