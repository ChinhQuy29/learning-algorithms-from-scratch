import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      
        self.threshold = threshold  
        self.left = left            
        self.right = right          
        self.value = value          

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        
    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini
    
    def _information_gain(self, y, y_left, y_right, criterion):
        if criterion == 'gini':
            impurity_func = self._gini
        else: 
            impurity_func = self._entropy
            
        parent_impurity = impurity_func(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
            
        child_impurity = (n_left / n) * impurity_func(y_left) + (n_right / n) * impurity_func(y_right)
        information_gain = parent_impurity - child_impurity
        
        return information_gain
    
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None, None, None, None
            
        best_gain = -np.inf
        best_feature, best_threshold = None, None
        best_left_indices, best_right_indices = None, None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gain = self._information_gain(
                    y, 
                    y[left_indices], 
                    y[right_indices],
                    self.criterion
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices
                    
        return best_feature, best_threshold, best_left_indices, best_right_indices, best_gain
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold, best_left_indices, best_right_indices, best_gain = self._best_split(X, y)
        
        if best_gain == -np.inf or best_gain == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_subtree = self._build_tree(
            X[best_left_indices], 
            y[best_left_indices], 
            depth + 1
        )
        
        right_subtree = self._build_tree(
            X[best_right_indices], 
            y[best_right_indices], 
            depth + 1
        )
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def _most_common_label(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
            
        if node.value is not None:
            print("  " * depth + f"Leaf: {node.value}")
            return
            
        print("  " * depth + f"Feature {node.feature} <= {node.threshold}")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)
