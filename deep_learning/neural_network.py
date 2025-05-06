import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize a neural network with specified layer sizes
        
        Parameters:
        - layer_sizes: list of integers, the size of each layer including input and output
        - learning_rate: learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(1, self.num_layers):
            # He initialization for weights
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2/layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def forward(self, X, activation_fn='sigmoid'):
        """
        Forward propagation
        
        Parameters:
        - X: input data, shape (n_samples, input_size)
        - activation_fn: activation function to use ('sigmoid' or 'relu')
        
        Returns:
        - activations: list of activations for each layer
        - z_values: list of weighted inputs for each layer
        """
        activations = [X]  # List to store activations of layers
        z_values = []      # List to store z values (weighted inputs)
        
        # Forward propagate through each layer
        for i in range(self.num_layers - 1):
            # Calculate weighted input
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function
            if activation_fn == 'sigmoid':
                activation = self.sigmoid(z)
            else:  # relu
                activation = self.relu(z)
            
            activations.append(activation)
            
        return activations, z_values
    
    def backward(self, X, y, activations, z_values, activation_fn='sigmoid'):
        """
        Backward propagation
        
        Parameters:
        - X: input data
        - y: target values
        - activations: list of activations from forward pass
        - z_values: list of weighted inputs from forward pass
        - activation_fn: activation function used ('sigmoid' or 'relu')
        
        Returns:
        - gradients for weights and biases
        """
        m = X.shape[0]  # Number of training examples
        
        # Initialize lists to store gradients
        dweights = [np.zeros_like(w) for w in self.weights]
        dbiases = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = activations[-1] - y
        
        # Backward pass through layers
        for l in reversed(range(self.num_layers - 1)):
            # Calculate gradients for weights and biases
            dweights[l] = np.dot(activations[l].T, delta) / m
            dbiases[l] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Calculate error for previous layer (if not the input layer)
            if l > 0:
                if activation_fn == 'sigmoid':
                    delta = np.dot(delta, self.weights[l].T) * self.sigmoid_derivative(activations[l])
                else:  # relu
                    delta = np.dot(delta, self.weights[l].T) * self.relu_derivative(activations[l])
        
        return dweights, dbiases
    
    def update_parameters(self, dweights, dbiases):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dweights[i]
            self.biases[i] -= self.learning_rate * dbiases[i]
    
    def train(self, X, y, epochs=1000, batch_size=None, activation_fn='sigmoid', verbose=True):
        """
        Train the neural network
        
        Parameters:
        - X: training data, shape (n_samples, input_size)
        - y: target values, shape (n_samples, output_size)
        - epochs: number of training epochs
        - batch_size: size of mini-batches (if None, use the whole dataset)
        - activation_fn: activation function to use ('sigmoid' or 'relu')
        - verbose: whether to print training progress
        """
        m = X.shape[0]
        
        for epoch in range(epochs):
            # Mini-batch gradient descent
            if batch_size is None:
                batch_size = m
            
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward pass
                activations, z_values = self.forward(X_batch, activation_fn)
                
                # Backward pass and update parameters
                dweights, dbiases = self.backward(X_batch, y_batch, activations, z_values, activation_fn)
                self.update_parameters(dweights, dbiases)
            
            # Print loss every 100 epochs if verbose
            if verbose and epoch % 100 == 0:
                activations, _ = self.forward(X, activation_fn)
                predictions = activations[-1]
                loss = np.mean((predictions - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X, activation_fn='sigmoid'):
        """Make predictions for input data"""
        activations, _ = self.forward(X, activation_fn)
        return activations[-1]
    
    def evaluate(self, X, y, activation_fn='sigmoid'):
        """Evaluate the model on test data"""
        predictions = self.predict(X, activation_fn)
        loss = np.mean((predictions - y) ** 2)
        return loss

# Example usage
if __name__ == "__main__":
    # Example: XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train neural network
    nn = NeuralNetwork([2, 4, 1], learning_rate=0.1)
    nn.train(X, y, epochs=10000, activation_fn='sigmoid')
    
    # Test predictions
    predictions = nn.predict(X)
    print("Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Predicted: {predictions[i][0]:.4f}, Actual: {y[i][0]}")
