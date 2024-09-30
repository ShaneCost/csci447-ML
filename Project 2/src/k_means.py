import math
import numpy as np
from hyperparameter import *

def distance(a, b):
    np_a = np.array(a)
    np_b = np.array(b)

    return np.linalg.norm(np_a - np_b)

def argmin(dictionary):
    return min(dictionary, key=dictionary.get)

class KMeans:
    def __init__(self, training_set):
        self.training_set = training_set

        self.num_features = len(training_set[0]) - 1

        self.centroids_locations = []
        self.centroid_set = []

        self.is_clustered = False

        self.num_clusters = Hyperparameter('num_clusters', math.ceil(math.sqrt(len(training_set))), 1)

        self.load()

    def load(self):
        self.tune_num_clusters()
        self.cluster()
        self.assign_centroid_values()

    def tune_num_clusters(self):
        pass

    def cluster(self):
        # Get the number of features (exclude the target or class column if present)
        training_without_class_or_target = np.array([x[:-1] for x in self.training_set], dtype=float)

        # Get the minimum and maximum values for each feature
        min_values = training_without_class_or_target.min(axis=0)
        max_values = training_without_class_or_target.max(axis=0)

        # Generate random initial centroid values
        mews = np.array([np.random.uniform(min_values, max_values) for _ in range(self.num_clusters.value)])

        while not self.is_clustered:
            centroids = {tuple(value): [] for value in mews}

            for x in training_without_class_or_target:
                distances = {}
                for value in mews:
                    distances[tuple(value)] = distance(x, value)

                centroids[argmin(distances)].append(x)

            new_mews = self.update_new(centroids)

            # Compare the old mews with new_mew
            if np.array_equal(mews, new_mews):
                self.is_clustered = True
                mews = new_mews
            else:
                mews = new_mews

        self.centroids_locations = mews

    def update_new(self, centroids):
        # Create a new NumPy array to store updated centroids, same shape as the initial mews
        new_mew = np.empty((self.num_clusters.value, len(self.training_set[0]) - 1))

        for i, (centroid, points) in enumerate(centroids.items()):
            if points:  # Avoid division by zero for empty clusters
                # Convert list of points to a NumPy array for easy mean calculation
                points_array = np.array(points)

                # Compute the mean of each feature (column) across all points assigned to the centroid
                new_centroid = points_array.mean(axis=0)

                # Update the new_mew array with the new centroid values
                new_mew[i] = new_centroid
            else:
                # If no points were assigned to the centroid, keep it the same
                new_mew[i] = np.array(centroid)

        return new_mew

    def assign_centroid_values(self):
        training = self.training_set







