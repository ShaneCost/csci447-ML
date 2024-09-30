import math
import numpy as np

def distance(a, b):
    np_a = np.array(a)
    np_b = np.array(b)

    return np.linalg.norm(np_a - np_b)


class KMeans:
    def __init__(self, training_set, test_set):
        self.training_set = training_set
        self.test_set = test_set

        self.num_clusters = math.ceil(math.sqrt(len(training_set)))
        self.centroids_locations = {}
        self.centroid_values = {}

    def tune_kc(self):
        pass

    def cluster(self):
        # Get the number of features (exclude the target or class column if present)
        training_without_class_or_target = np.array([x[:-1] for x in self.training_set], dtype=float)

        # Get the minimum and maximum values for each feature
        min_values = training_without_class_or_target[:, :-1].min(axis=0)
        max_values = training_without_class_or_target[:, :-1].max(axis=0)

        # Generate random initial centroid values
        mews = np.array([np.random.uniform(min_values, max_values) for _ in range(self.num_clusters)])

        for x in training_without_class_or_target:
            for mew in mews:
                dist = distance(x, mew)  # Only compare features, not the target/class


