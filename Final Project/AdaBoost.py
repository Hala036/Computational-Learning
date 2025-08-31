import numpy as np
from decision_tree import DecisionTree

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.alphas = []     # Model weights
        self.models = []     # Weak learners (trees)

    def fit(self, X, y):
        # ADDED: ensure arrays & shapes are consistent
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()             # 1D labels for internal processing
        y_col = y.reshape(-1, 1)              # CHANGED: your DecisionTree.fit concatenates -> needs (n,1)

        # ADDED: convert labels to {-1, +1} for correct AdaBoost math
        y_bin = np.where(y == 1, 1.0, -1.0)

        n_samples = X.shape[0]

        # Initialize weights equally
        sample_weights = np.ones(n_samples, dtype=float) / n_samples

        # Reset containers
        self.alphas = []
        self.models = []

        for _ in range(self.n_estimators):
            # Train weak learner (stump = max_depth=1)
            # ADDED: force a stump and minimum split so it behaves like a weak learner
            model = DecisionTree(max_depth=1, min_samples_split=2)
            # CHANGED: pass y as a column vector to match your DecisionTree.fit
            model.fit(X, y_col, sample_weights=sample_weights)

            # Predict on training data
            y_pred = np.asarray(model.predict(X))
            # ADDED: convert predictions to {-1, +1}
            h_bin = np.where(y_pred == 1, 1.0, -1.0)

            # Weighted classification error: sum of weights where prediction != label
            miss = (h_bin != y_bin)
            error = np.sum(sample_weights[miss])

            # ADDED: clamp error to (0,1) to avoid infinities / zero division
            eps = 1e-12
            error = float(np.clip(error, eps, 1 - eps))

            # Compute model weight (alpha)
            alpha = 0.5 * np.log((1.0 - error) / error)

            # ADDED: update sample weights using AdaBoost rule w_i *= exp(-alpha * y_i * h_i)
            sample_weights *= np.exp(-alpha * y_bin * h_bin)
            sample_weights /= np.sum(sample_weights)  # normalize

            # Save learner & its weight
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        X = np.asarray(X, dtype=float)  # ADDED: ensure numeric array

        # ADDED: handle case where fit() added no models (edge case)
        if not self.models:
            return np.zeros(X.shape[0], dtype=int)

        # Weighted vote of all models in {-1, +1} space
        F = np.zeros(X.shape[0], dtype=float)
        for alpha, model in zip(self.alphas, self.models):
            pred = np.asarray(model.predict(X))
            h_bin = np.where(pred == 1, 1.0, -1.0)  # Convert to ±1
            F += alpha * h_bin

        # ADDED: convert sign(F) back to {0,1}; tie (0) favors +1 class
        y_hat_bin = np.sign(F)
        y_hat_bin[y_hat_bin == 0] = 1.0
        return np.where(y_hat_bin == 1.0, 1, 0)
