import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None ):
        """Constructor"""
        # decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain # information gain: uses entropy and node weight to decide which split to choose

        # leaf node
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        """Constructor"""
        # initialise root of tree
        self.root = None

        # Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0, sample_weights=None):
        """ Recursive function to build the tree with sample weights """

        if curr_depth is None:
            curr_depth = 0

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape

        # Check stopping conditions [DEBUG]
        if num_samples >= self.min_samples_split and (curr_depth <= self.max_depth):
            # Find best split using weights
            best_split = self.get_best_split(dataset, num_samples, num_features, sample_weights)

            # If we found a valid split
            if best_split["info_gain"] > 0:
                # Split weights accordingly
                left_weights = right_weights = None
                if sample_weights is not None:
                    left_indices = best_split["indices_left"]
                    right_indices = best_split["indices_right"]
                    if left_indices is not None and right_indices is not None:
                        left_weights = sample_weights[left_indices]
                        right_weights = sample_weights[right_indices]

                # Recursive calls
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1, left_weights)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1, right_weights)

                return Node(
                    feature_index=best_split["feature_index"], # the inquery looks like: feature_index <= threshold
                    threshold=best_split["threshold"],
                    left=left_subtree,
                    right=right_subtree,
                    info_gain=best_split["info_gain"]
                )

        # If we're at a leaf
        leaf_value = self.calculate_leaf_value(Y, sample_weights)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features, sample_weights=None):
        """ function to find the best split """

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        y = dataset[:, -1]

        # Loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                # check if children are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    left_y = dataset_left[:, -1]
                    right_y = dataset_right[:, -1]

                    # compute information gain
                    # Compute corresponding weights (if any)
                    if sample_weights is not None:
                        indices = np.arange(len(dataset))
                        left_mask = dataset[:, feature_index] <= threshold
                        right_mask = ~left_mask

                        left_weights = sample_weights[left_mask]
                        right_weights = sample_weights[right_mask]

                        curr_info_gain = self.weighted_information_gain(
                            y, left_y, right_y, "gini",
                            sample_weights=sample_weights,
                            left_weights=left_weights,
                            right_weights=right_weights
                        )
                    else:
                        # use gini because it has a better runtime
                        curr_info_gain = self.information_gain(
                            y, left_y, right_y, "gini"
                        )

                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain

                        # Store masks to later retrieve sample_weight splits in build_tree
                        if sample_weights is not None:
                            best_split["indices_left"] = np.where(left_mask)[0]
                            best_split["indices_right"] = np.where(right_mask)[0]

                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        """ function to split the data """

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def weighted_information_gain(self, parent, l_child, r_child, mode="entropy", sample_weights=None, left_weights=None, right_weights=None):
        """ function to compute information gain with sample weights """

        # Total weight of parent node
        total_weight = np.sum(sample_weights)

        # Weights of the child nodes
        weight_l = np.sum(left_weights) / total_weight
        weight_r = np.sum(right_weights) / total_weight

        if mode == "gini":
            gain = self.weighted_gini_index(parent, sample_weights) - (
                    weight_l * self.weighted_gini_index(l_child, left_weights) +
                    weight_r * self.weighted_gini_index(r_child, right_weights)
            )
        else:
            gain = self.weighted_entropy(parent, sample_weights) - (
                    weight_l * self.weighted_entropy(l_child, left_weights) +
                    weight_r * self.weighted_entropy(r_child, right_weights)
            )

        return gain

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        """ function to compute information gain """

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    # weighted entropy used when sample weights are provided by the calling class (AdaBoost)
    def weighted_entropy(self, y, weights):
        """Weighted entropy = - sum_i (w_i / W) * log2(w_i / W) for each class"""
        class_labels = np.unique(y)
        total_weight = np.sum(weights)
        entropy = 0

        for cls in class_labels:
            mask = (y == cls)
            class_weight = np.sum(weights[mask])  # sum of weights for this class
            if class_weight > 0:
                p_cls = class_weight / total_weight
                entropy += -p_cls * np.log2(p_cls)
        return entropy

    # unweighted entropy used when no sample weights are provided (Random Forest)
    def entropy(self, y):
        """ function to compute entropy """
        """ entropy = - sum(i=1 to 2)[pi * log_2(pi)]"""

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y) # pi
            entropy += -p_cls * np.log2(p_cls) # -pi * log_2(pi)
        return entropy

    # weighted Gini index used when sample weights are provided by the calling class (AdaBoost)
    def weighted_gini_index(self, y, weights):
        """Weighted Gini = 1 - sum_i (w_i / W)^2"""
        class_labels = np.unique(y)
        total_weight = np.sum(weights)
        gini = 0

        for cls in class_labels:
            mask = (y == cls)
            class_weight = np.sum(weights[mask])
            if class_weight > 0:
                p_cls = class_weight / total_weight
                gini += p_cls ** 2
        return 1 - gini

    # unweighted Gini index used when no sample weights are provided (Random Forest)
    def gini_index(self, y):
        """ function to compute gini index """
        """ gini = 1 - [ sum(i=1 to 2)[pi^2] ]"""

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y) # pi
            gini += p_cls ** 2 # pi^2
        return 1 - gini

    def calculate_leaf_value(self, Y, sample_weights=None):
        """Compute the leaf node value using majority vote, optionally weighted."""
        class_labels = np.unique(Y)
        if sample_weights is None:
            # Use majority class
            Y = list(Y)
            return max(Y, key=Y.count)
        else:
            # Use weighted majority
            weights_by_class = {cls: np.sum(sample_weights[Y == cls]) for cls in class_labels}
            return max(weights_by_class, key=weights_by_class.get)

    def print_tree(self, tree=None, indent=" "):
        """ function to print the tree """

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y, sample_weights=None):
        """ function to train the tree """

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset, curr_depth=0, sample_weights=sample_weights)

    def predict(self, X):
        """ function to predict new dataset """

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        """ function to predict a single data point """

        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)