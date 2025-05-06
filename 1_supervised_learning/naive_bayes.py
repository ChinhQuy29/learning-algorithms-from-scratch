import numpy as np
from collections import Counter

class NaiveBayes:
    def __init__(self):
        self.prior_probabilities = {} 
        self.likelihood = {}  
        self.training_examples = None
        self.training_labels = None

    def fit(self, X, y):
        self.training_examples = X
        self.training_labels = y
        labels_counter = Counter(y)
        labels = list(labels_counter.keys())

        num_samples = len(y)
        num_features = len(X[0])
        
        for label in labels:
            self.prior_probabilities[label] = labels_counter[label] / num_samples
            self.likelihood[label] = {i: {} for i in range(num_features)}

        for label in labels:
            label_indices = [i for i in range(num_samples) if y[i] == label]
            label_feature_values = np.array([X[i] for i in label_indices]).T 

            for feature_idx in range(num_features):
                feature_counts = Counter(label_feature_values[feature_idx])
                total_label_samples = len(label_indices)
                unique_feature_values = len(feature_counts)

                for feature_value, count in feature_counts.items():
                    self.likelihood[label][feature_idx][feature_value] = (count + 1) / (total_label_samples + unique_feature_values)

    def predict(self, X):
        predictions = []
        
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
