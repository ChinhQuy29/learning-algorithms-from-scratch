import math
from collections import Counter

class KNearestNeighbors():
    def __init__(self, k=5, distance_function=0, p=2):
        self.k = k
        self.distance_function = distance_function
        self.p = p
        self.training_examples = None
        self.training_labels = None

    def euclidean_distance(self, x1, x2):
        num_features = len(x1)
        return math.sqrt(sum((x1[i] - x2[i]) ** 2 for i in range(num_features)))

    def manhattan_distance(self, x1, x2):
        num_features = len(x1)
        return sum(abs(x1[i] - x2[i]) for i in range(num_features))

    def minkowski_distance(self, x1, x2, p):
        num_features = len(x1)
        return sum(abs(x1[i] - x2[i]) ** p for i in range(num_features)) ** (1 / p)

    def hamming_distance(self, x1, x2):
        num_features = len(x1)
        return sum(1 for i in range(num_features) if x1[i] != x2[i])

    def fit(self, X, y):
        self.training_examples = X
        self.training_labels = y

    def find_neighbors(self, X, distance_function):
        distances = []
        for i in range(len(self.training_examples)):
            distance = distance_function(X, self.training_examples[i])
            distances.append((distance, self.training_labels[i]))
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

    def predict(self, X):
        predictions = []
        for x in X:
            if self.distance_function == 0:
                neighbors = self.find_neighbors(x, self.euclidean_distance)
            elif self.distance_function == 1:
                neighbors = self.find_neighbors(x, self.manhattan_distance)
            elif self.distance_function == 2:
                neighbors = self.find_neighbors(x, lambda a, b: self.minkowski_distance(a, b, self.p))
            elif self.distance_function == 3:
                neighbors = self.find_neighbors(x, self.hamming_distance)
            else:
                raise ValueError("Invalid distance function")

            labels = [label for _, label in neighbors]
            most_common = Counter(labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions

