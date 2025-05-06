import numpy as np  

class LogisticRegression():
    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = []
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        m = len(X)
        n = len(X[0])
        cost = 0
        for i in range(m):
            z = sum([self.weights[j] * X[i][j] for j in range(n)]) + self.bias
            prediction = self.sigmoid(z)
            cost += - (y[i] * np.log(prediction) + (1 - y[i]) * np.log(1 - prediction)) / m
        return cost
    
    def compute_gradients(self, X, y):
        m = len(X)
        n = len(X[0])
        dj_dw = [0] * n
        dj_db = 0
        for i in range(m):
            z = sum([self.weights[j] * X[i][j] for j in range(n)]) + self.bias
            prediction = self.sigmoid(z)
            error = prediction - y[i]
            for j in range(n):
                dj_dw[j] += error * X[i][j] / m
            dj_db += error / m
        return dj_dw, dj_db

    def fit(self, X, y):
        self.weights = [0] * len(X[0])
        m = len(X)
        n = len(X[0])
        for i in range(self.num_iterations):
            cost = self.compute_cost(X, y)
            dj_dw, dj_db = self.compute_gradients(X, y)
            for j in range(n):
                self.weights[j] -= self.learning_rate * dj_dw[j]
            self.bias -= self.learning_rate * dj_db
            print(f"Iteration {i+1}/{self.num_iterations}, Cost: {cost}")

    def predict(self, X):
        predictions = []
        m = len(X)
        n = len(X[0])
        for i in range(m):
            z = sum([self.weights[j] * X[i][j] for j in range(n)]) + self.bias
            prediction = self.sigmoid(z)
            predictions.append(1 if prediction >= 0.5 else 0)
        return predictions

