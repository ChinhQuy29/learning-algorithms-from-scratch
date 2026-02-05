import numpy as np

class SoftmaxRegression:
    # Initializing the model
    def __init__(self, input_size, num_classes, learning_rate=0.01):
        self.W = np.random.randn(input_size, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        self.learning_rate = learning_rate

    # Softmax function
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Cross-Entropy Loss
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    # Training the model
    def fit(self, X, y, epochs=1000):
        m = X.shape[0]
        
        for epoch in range(epochs):
            # Forward pass
            logits = np.dot(X, self.W) + self.b
            y_pred = self.softmax(logits)
            loss = self.compute_loss(y, y_pred)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
            # Backward pass
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[np.arange(m), y] = 1
            dZ = y_pred - y_true_one_hot
            dW = np.dot(X.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            # Update weights and bias
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    # Making predictions
    def predict(self, X):
        logits = np.dot(X, self.W) + self.b
        y_pred = self.softmax(logits)
        return np.argmax(y_pred, axis=1)
    
    # Evaluating the model
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = SoftmaxRegression(input_size=X.shape[1], num_classes=len(np.unique(y)), learning_rate=0.1)
    model.fit(X_train, y_train, epochs=1000)

    # Evaluate model
    accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')