class ElasticNet():
    def __init__(self, learning_rate=0.01, num_iterations=1000, ridge_param=0.01, lasso_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.ridge_param = ridge_param
        self.lasso_param = lasso_param
        self.weights = []
        self.bias = 0

    def compute_cost(self, X, y):
        num_examples = len(X)
        num_features = len(X[0])
        cost = 0
        for i in range(num_examples):
            prediction = sum([self.weights[j] * X[i][j] for j in range(num_features)]) + self.bias
            error = prediction - y[i]
            cost += error ** 2
        ridge_penalty = sum([weight ** 2 for weight in self.weights]) * self.ridge_param / (2 * num_examples)
        lasso_penalty = sum([abs(weight) for weight in self.weights]) * self.lasso_param / (2 * num_examples)
        return cost / (2 * num_examples) + ridge_penalty + lasso_penalty

    def compute_gradients(self, X, y):
        num_examples = len(X)
        num_features = len(X[0])
        dj_dw = [0] * num_features
        dj_db = 0
        for i in range(num_examples):
            prediction = sum([self.weights[j] * X[i][j] for j in range(num_features)]) + self.bias
            error = prediction - y[i]
            for j in range(num_features):
                dj_dw[j] += error * X[i][j] / num_examples
            dj_db += error / num_examples
        for j in range(num_features):
            if (self.weights[j] >= 0):
                dj_dw[j] += self.ridge_param / num_examples * self.weights[j] + self.lasso_param / (2 * num_examples)  
            else:
                dj_dw[j] += self.ridge_param / num_examples * self.weights[j] - self.lasso_param / (2 * num_examples)
        return dj_dw, dj_db

    def fit(self, X, y):
        num_features = len(X[0])
        self.weights = [0] * num_features
        for i in range(self.num_iterations):
            dj_dw, dj_db = self.compute_gradients(X, y)
            for j in range(num_features):
                self.weights[j] -= self.learning_rate * dj_dw[j]
            self.bias -= self.learning_rate * dj_db
            cost = self.compute_cost(X, y)
            print(f"Iteration {i+1}/{self.num_iterations}, Cost: {cost}")

    def predict(self, X):
        predictions = []
        num_examples = len(X)
        num_features = len(X[0])
        for i in range(num_examples):
            prediction = sum([self.weights[j] * X[i][j] for j in range(num_features)]) + self.bias
            predictions.append(prediction)
        return predictions

    