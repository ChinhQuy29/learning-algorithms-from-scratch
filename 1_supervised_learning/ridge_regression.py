class RidgeRegression():
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_param=0.01):
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
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
        ridge_penalty = sum([self.weights[j] ** 2 * self.regularization_param for j in range(num_features)])
        return (cost + ridge_penalty) / (2 * num_examples)

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
            dj_dw[j] += self.regularization_param / num_examples * self.weights[j]
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
        num_examples = len(X)
        num_features = len(X[0])
        predictions = []
        for i in range(num_examples):
            prediction = sum([self.weights[j] * X[i][j] for j in range(num_features)]) + self.bias
            predictions.append(prediction)
        return predictions
        
