# Gaussian Naive Bayes Classifier Implementation
import numpy as np
class GaussianNaiveBayes:
    # Initialize the classifier
    def __init__(self):
        self.classes = None
        self.means = None
        self.variances = None
        self.priors = None

    # Fit the model to the training data
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        # Calculate means, variances, and priors for each class
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.means[idx, :] = X_cls.mean(axis=0)
            self.variances[idx, :] = X_cls.var(axis=0)
            self.priors[idx] = X_cls.shape[0] / n_samples

    # Calculate Gaussian probability density function
    def _gaussian_probability(self, class_idx, x):
        # Calculate the Gaussian probability for each feature
        mean = self.means[class_idx]
        variance = self.variances[class_idx]
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent

    # Predict the class labels for the input data
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for idx, cls in enumerate(self.classes):
            prior_log = np.log(self.priors[idx])
            likelihood_log = np.sum(np.log(self._gaussian_probability(idx, X)), axis=1)
            log_probs[:, idx] = prior_log + likelihood_log

        return self.classes[np.argmax(log_probs, axis=1)]

    # Evaluate the model's accuracy
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
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # Evaluate model
    accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')