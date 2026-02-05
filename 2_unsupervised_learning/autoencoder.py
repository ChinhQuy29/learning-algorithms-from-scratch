import numpy as np
 
class Autoencoder:
    # Initializing the model
    def __init__(self, input_size, hidden_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, input_size) * 0.01
        self.b2 = np.zeros((1, input_size))

    #  Encode the input
    def encode(self, X):
        return np.tanh(np.dot(X, self.W1) + self.b1)
    
    # Decode the encoded representation
    def decode(self, h):
        return self.sigmoid(np.dot(h, self.W2) + self.b2)
    
    # Mean Squared Error Loss
    def compute_loss(self, X, X_recon):
        return np.mean((X - X_recon) ** 2)
    
    # Training the model
    def fit(self, X, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            h = self.encode(X)
            X_recon = self.decode(h)
            loss = self.compute_loss(X, X_recon)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
            # Backward pass
            dX_recon = -(X - X_recon) / X.shape[0]
            dW2 = np.dot(h.T, dX_recon * X_recon * (1 - X_recon))
            db2 = np.sum(dX_recon * X_recon * (1 - X_recon), axis=0, keepdims=True)
            dh = np.dot(dX_recon * X_recon * (1 - X_recon), self.W2.T) * (1 - h ** 2)
            dW1 = np.dot(X.T, dh)
            db1 = np.sum(dh, axis=0, keepdims=True)
            # Update weights and biases
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) 
    
    # Reconstruct input data
    def reconstruct(self, X):
        h = self.encode(X)
        X_recon = self.decode(h)
        return X_recon
    
    # Get encoded representation
    def get_encoded_representation(self, X):
        return self.encode(X)
    
# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_digits()
    X = data.data / 16.0  # Normalize pixel values to [0, 1]

    # Split dataset
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Initialize and train autoencoder
    autoencoder = Autoencoder(input_size=64, hidden_size=32, learning_rate=0.01)
    autoencoder.fit(X_train, epochs=1000)

    # Reconstruct test data
    X_recon = autoencoder.reconstruct(X_test)

    # Print original and reconstructed samples
    import matplotlib.pyplot as plt

    n = 10  # Number of samples to display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(X_recon[i].reshape(8, 8), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()