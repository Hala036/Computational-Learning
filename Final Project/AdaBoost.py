import numpy as np
from decision_tree import DecisionTree

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.alphas = []     # Model weights
        self.models = []     # Weak learners (stumps/tree)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel() # we used ravel to make y a 1-dimensional array instead of 2D
        y_col = y.reshape(-1, 1) # the decision tree fit method expects a column vector, so we created this variable to provide it

        # transform 0/1 classes to -1/1, in order for the calculations used in AdaBoost to be optimal
        y_bin = np.where(y == 1, 1.0, -1.0)

        n_samples = X.shape[0]

        # Initialize weights equally
        sample_weights = np.ones(n_samples, dtype=float) / n_samples

        # Reset containers
        self.alphas = []
        self.models = []

        for _ in range(self.n_estimators):
            # for each iteration, create a new decision tree with max depth 1
            model = DecisionTree(max_depth=1, min_samples_split=2)
            # run DecisionTree.fit, with the sample weights so that each model can focus on the intended samples
            # (preciously misclassified samples)
            model.fit(X, y_col, sample_weights=sample_weights)

            # Predict on training data
            y_pred = np.asarray(model.predict(X))
            h_bin = np.where(y_pred == 1, 1.0, -1.0)

            # we calculate the error by checking how many misclassification we got and summing them together,
            # then dividing by the total samples weight
            miss = (h_bin != y_bin)
            error = np.sum(sample_weights[miss]) / np.sum(sample_weights)

            # place error in the range 0-1 to prevent division by 0 in case err=0, or log(0) in case err=1
            eps = 1e-12
            error = float(np.clip(error, eps, 1 - eps))

            # Compute model weight (alpha)~ determines the importance of this model
            alpha = 0.5 * np.log((1.0 - error) / error)

            # update sample weights: w[i] *= exp(alpha * y[i] * h[i]) / (-alpha for correctly classified samples)
            # this way we give higher weights to misclassified samples so that the next model can focus on them
            sample_weights *= np.exp(-alpha * y_bin * h_bin)
            sample_weights /= np.sum(sample_weights)  # normalize

            # Save learner and its weight
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        # return all zeros if no models were initialised and trained
        if not self.models:
            return np.zeros(X.shape[0], dtype=int)

        # Weighted vote of all models in {-1, +1} space
        # F is the array that saves in each index, each models weight multiplied by its prediction
        F = np.zeros(X.shape[0], dtype=float)
        for alpha, model in zip(self.alphas, self.models):
            pred = np.asarray(model.predict(X))
            h_bin = np.where(pred == 1, 1.0, -1.0)  # Convert to ±1
            F += alpha * h_bin

        # create the array 'y_hat_bin' that has +1 for values that are ≥0 and -1 for negative values in F
        y_hat_bin = np.sign(F)
        y_hat_bin[y_hat_bin == 0] = 1.0
        # convert the -1's to 0 (our original classes)
        return np.where(y_hat_bin == 1.0, 1, 0)
