import numpy as np
from collections import Counter

class NaiveBayes:
    # Initializing the model
    def __init__(self):
        self.prior_probabilities = {} 
        self.likelihood = {}  
        self.training_examples = None
        self.training_labels = None

    # Training the model
    def fit(self, X, y):
        self.training_examples = X
        self.training_labels = y
        labels_counter = Counter(y)
        labels = list(labels_counter.keys())

        num_samples = len(y)
        num_features = len(X[0])
        
        # Calculating prior probabilities and likelihoods
        for label in labels:
            self.prior_probabilities[label] = labels_counter[label] / num_samples
            self.likelihood[label] = {i: {} for i in range(num_features)}

        # Calculating likelihoods with Laplace smoothing
        for label in labels:
            label_indices = [i for i in range(num_samples) if y[i] == label]
            label_feature_values = np.array([X[i] for i in label_indices]).T 

            for feature_idx in range(num_features):
                feature_counts = Counter(label_feature_values[feature_idx])
                total_label_samples = len(label_indices)
                unique_feature_values = len(feature_counts)

                for feature_value, count in feature_counts.items():
                    self.likelihood[label][feature_idx][feature_value] = (count + 1) / (total_label_samples + unique_feature_values)

    # Making predictions
    def predict(self, X):
        predictions = []
        # Calculating posterior probabilities and making predictions
        for example in X:
            posteriors = {}

            for label in self.prior_probabilities:
                posterior = self.prior_probabilities[label]

                for feature_idx, feature_value in enumerate(example):
                    if feature_value in self.likelihood[label][feature_idx]:
                        posterior *= self.likelihood[label][feature_idx][feature_value]
                    else:
                        num_total_samples = len(self.training_examples)
                        num_unique_values = len(self.likelihood[label][feature_idx])
                        posterior *= 1 / (num_total_samples + num_unique_values)

                posteriors[label] = posterior

            predictions.append(max(posteriors, key=posteriors.get))

        return predictions
    
    # Calculating accuracy
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(np.array(predictions) == np.array(y))

# Example usage:
if __name__ == "__main__":
    # Sample dataset
    X_train = [
        [1, 'Sunny', 'Hot', 'High', False],
        [2, 'Sunny', 'Hot', 'High', True],
        [3, 'Overcast', 'Hot', 'High', False],
        [4, 'Rain', 'Mild', 'High', False],
        [5, 'Rain', 'Cool', 'Normal', False],
        [6, 'Rain', 'Cool', 'Normal', True],
        [7, 'Overcast', 'Cool', 'Normal', True],
        [8, 'Sunny', 'Mild', 'High', False],
        [9, 'Sunny', 'Cool', 'Normal', False],
        [10, 'Rain', 'Mild', 'Normal', False],
        [11, 'Sunny', 'Mild', 'Normal', True],
        [12, 'Overcast', 'Mild', 'High', True],
        [13, 'Overcast', 'Hot', 'Normal', False],
        [14, 'Rain', 'Mild', 'High', True]
    ]
    y_train = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
               'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

    X_test = [
        [15, 'Sunny', 'Cool', 'High', True],
        [16, 'Overcast', 'Mild', 'Normal', False],
        [17, 'Rain', 'Hot', 'High', True]
    ]
    y_test = ['No', 'Yes', 'No']

    # Create and train the Naive Bayes classifier
    nb_classifier = NaiveBayes()
    nb_classifier.fit(X_train, y_train)

    # Make predictions
    predictions = nb_classifier.predict(X_test)
    print("Predictions:", predictions)

    # Calculate accuracy
    accuracy = nb_classifier.score(X_test, y_test)
    print("Accuracy:", accuracy)
