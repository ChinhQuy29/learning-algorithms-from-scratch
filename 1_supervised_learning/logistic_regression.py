import numpy as np  

class LogisticRegression():
    # Initializing the model
    def __init__(self, learning_rate = 0.001, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = []
        self.bias = 0

    # Sigmoid function
    def sigmoid(self, z):
        # Numerically stable sigmoid to prevent overflow
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        exp_z = np.exp(z)
        return exp_z / (1 + exp_z)
    
    # Binary Cross-Entropy Cost Function
    def compute_cost(self, X, y):
        m = len(X)
        n = len(X[0])
        cost = 0
        eps = 1e-15 # to avoid log(0)
        for i in range(m):
            z = sum([self.weights[j] * X[i][j] for j in range(n)]) + self.bias
            prediction = self.sigmoid(z)
            prediction = np.clip(prediction, eps, 1 - eps) # clip prediction
            cost += - (y[i] * np.log(prediction) + (1 - y[i]) * np.log(1 - prediction)) / m
        return cost
    
    # Gradient Descent
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

    # Training the model
    def fit(self, X, y):
        self.weights = [0] * len(X[0])
        self.bias = 0
        m = len(X)
        n = len(X[0])
        for i in range(self.num_iterations):
            cost = self.compute_cost(X, y)
            dj_dw, dj_db = self.compute_gradients(X, y)
            for j in range(n):
                self.weights[j] -= self.learning_rate * dj_dw[j]
            self.bias -= self.learning_rate * dj_db
            if i % 100 == 0:
                print(f"Iteration {i+1}/{self.num_iterations}, Cost: {cost}")

    # Making predictions
    def predict(self, X):
        predictions = []
        m = len(X)
        n = len(X[0])
        for i in range(m):
            z = sum([self.weights[j] * X[i][j] for j in range(n)]) + self.bias
            prediction = self.sigmoid(z)
            predictions.append(1 if prediction >= 0.5 else 0)
        return predictions
    
# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train.tolist(), y_train.tolist())

    # Make predictions
    predictions = model.predict(X_test.tolist())

    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

