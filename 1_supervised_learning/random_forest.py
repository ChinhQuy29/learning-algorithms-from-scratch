import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True, criterion='gini'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_max_features(self, n_features):
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            return n_features  # default: use all features
    
    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)
        
        for _ in range(self.n_trees):
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion
            )
            
            feature_indices = np.random.choice(n_features, size=max_features, replace=False)
            X_subset = X_sample[:, feature_indices]

            tree.fit(X_subset, y_sample)

            self.trees.append((tree, feature_indices))
            
        return self
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)

        final_predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            values, counts = np.unique(predictions[i, :], return_counts=True)
            final_predictions[i] = values[np.argmax(counts)]
            
        return final_predictions
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def feature_importance(self, X, y):
        base_score = self.score(X, y)
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
            permuted_score = self.score(X_permuted, y)
            importances[feature_idx] = base_score - permuted_score
            
        return importances
