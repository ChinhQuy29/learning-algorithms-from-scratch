class LinearRegression(): 
    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = []
        self.bias = 0

    def compute_cost(self, X, y):
        m = len(X)
        cost = 0
        for i in range(m):
            prediction = sum([self.weights[j] * X[i][j] for j in range(len(X[0]))]) + self.bias
            error = prediction - y[i]
            cost += error ** 2
        return cost / (2 * m)

    def compute_gradients(self, X, y):
        dj_dw = [0] * len(X[0])
        dj_db = 0
        m = len(X)
        for i in range(m):
            prediction = sum([self.weights[j] * X[i][j] for j in range(len(X[0]))]) + self.bias
            error = prediction - y[i]
            for j in range(len(X[0])):
                dj_dw[j] += error * X[i][j] / m
            dj_db += error / m
        return dj_dw, dj_db
    
    def fit(self, X, y):
        m = len(X)
        self.weights = [0] * len(X[0])
        for i in range(self.num_iterations):
            cost = self.compute_cost(X, y)
            dj_dw, dj_db = self.compute_gradients(X, y)
            for j in range(len(self.weights)):
                self.weights[j] -= self.learning_rate * dj_dw[j]
            self.bias -= self.learning_rate * dj_db
            print(f"Iteration {i+1}/{self.num_iterations}, Cost: {cost}")
        
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            prediction = sum([self.weights[j] * X[i][j] for j in range(len(X[0]))]) + self.bias
            predictions.append(prediction)
        return predictions


X = [[1], [2], [3], [4], [5], [6]]
y = [2, 4, 6, 8, 10, 12]

model = LinearRegression(learning_rate=0.01, num_iterations=10)

model.fit(X, y)

predictions = model.predict([[7]])

print(f"y = {model.weights[0]} * x + {model.bias}")

print("Predictions:", predictions)

