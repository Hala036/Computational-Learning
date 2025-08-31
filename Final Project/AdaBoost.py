import numpy as np
from decision_tree import DecisionTree

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.alphas = []     # Model weights
        self.models = []     # Weak learners (trees)

    def fit(self, X, y):
        y_transformed = np.where(y.ravel() == 0, -1, 1)
        n_samples = X.shape[0]
        # Initialize weights equally
        sample_weights = np.ones(n_samples) / n_samples

        self.alphas = []
        self.models = []

        for _ in range(self.n_estimators):
            # Train weak learner (stump = max_depth=1)
            model = DecisionTree(max_depth=1)
            model.fit(X, y, sample_weights=sample_weights)

            # Predict and compute error
            y_pred = model.predict(X)
            y_pred_transformed = np.where(y_pred == 0, -1, 1)

            # Weighted error
            error = np.sum(sample_weights[y_pred_transformed != y_transformed])

            # Stop if error too high or zero
            if error > 0.5 or error == 0:
                break

            # Compute model weight (alpha)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update weights: increase weight for misclassified samples
            sample_weights *= np.exp(-alpha * y_transformed * y_pred_transformed)
            sample_weights /= np.sum(sample_weights)  # Normalize

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # Weighted vote of all models
        final_pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            pred = model.predict(X)
            pred = np.where(pred == 1, 1, -1)  # Convert labels to ±1
            final_pred += alpha * pred

        return np.where(final_pred >= 0, 1, 0)  # Convert back to 0/1