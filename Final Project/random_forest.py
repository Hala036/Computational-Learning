import numpy as np
from collections import Counter
from decision_tree import DecisionTree  # Make sure this file is in the same folder

class RandomForest:
    def __init__(self, n_estimators=10, min_samples_split=2, max_depth=2, max_features=None):
        self.n_estimators = n_estimators #numbers of trees to build
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features  # number of features to sample at each split
        self.trees = []

    #Trains the forest. It starts by resetting the tree list.
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]

            # Random selection of features
            if (self.max_features is not None) and (self.max_features <= len(X_sample)):
                feature_indices = np.random.choice(X.shape[1], size=self.max_features, replace=False)
            else:
                feature_indices = np.arange(X.shape[1])

            X_sample_subset = X_sample[:, feature_indices]

            # Train a decision tree
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X_sample_subset, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees]) # each row is one tree's predictions for all given samples
        # Transpose: shape becomes (n_samples, n_trees)
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # Majority vote
        y_pred = [self._most_common_label(row) for row in tree_preds] # each row is an array of all predictions for that sample
        return np.array(y_pred)

    def _most_common_label(self, labels):
        counter = Counter(labels)
        return counter.most_common(1)[0][0]