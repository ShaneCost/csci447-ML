__author__ = "<Shane Costello>"

import numpy as np
from knn import *

def distance(a, b):
    """
    Static function used to calculate the euclidian distance between vectors

    :param a: A data point from the training set
    :param b: A centroid

    :return: The distance between a and b
    """
    np_a = np.array(a)
    np_b = np.array(b)

    return np.linalg.norm(np_a - np_b)

def argmin(dictionary):
    """
    A static function used to find the index of the minimum value in a dictionary

    :param dictionary: A dictionary of distances
    :return: The index of the smallest distance in the dictionary
    """
    return min(dictionary, key=dictionary.get)

class KMeans:
    def __init__(self, training_set, num_clusters, class_or_regress, print_cluster=False):
        """
        Class used to implement k-means clustering algorithm. Used to derive a reduced data set

        :param training_set: data set to be clustered
        :param num_clusters: integer of k clusters to be created
        :param class_or_regress: string denoting whether it is a classification or regression problem
        """
        self.training_set = training_set

        self.num_features = len(training_set[0]) - 1

        self.centroids_locations = []
        self.centroid_set = []

        self.is_clustered = False

        self.num_clusters = num_clusters

        if class_or_regress == 'class':
            self.is_class = True
        else:
            self.is_class = False

        self.print = print_cluster

        self.load()

    def load(self):
        """
        Function called inside constructor to run program

        :return: None
        """
        self.cluster()
        self.assign_centroid_values()

    def cluster(self):
        """
        Function used to create k-clusters

        :return: Set of centroid locations
        """
        # Get all feature vectors (exclude the target or class column)
        training_without_class_or_target = np.array([x[:-1] for x in self.training_set], dtype=float)

        # Get the minimum and maximum values for each feature
        min_values = training_without_class_or_target.min(axis=0)
        max_values = training_without_class_or_target.max(axis=0)

        # Generate random initial centroid locations within the min and max values
        mews = np.array([np.random.uniform(min_values, max_values) for _ in range(self.num_clusters)])

        # Iterate until clustered
        while not self.is_clustered:
            centroids = {tuple(value): [] for value in mews}

            for x in training_without_class_or_target:
                distances = {}
                for value in mews:
                    distances[tuple(value)] = distance(x, value)

                centroids[argmin(distances)].append(x)
                if self.print:
                    self.print_point_to_cluster(x.tolist(), distances)
                    self.print = False

            new_mews = self.update_new(centroids)

            # Compare the old mews with new_mew
            if np.array_equal(mews, new_mews):
                self.is_clustered = True
                mews = new_mews
            else:
                mews = new_mews

        self.centroids_locations = mews


    def update_new(self, centroids):
        """
        Function used to update centroid locations based on the mean of all points assigned to it's cluster

        :param centroids: set of centroids and all data points assigned to it's cluster
        :return: New centroid locations
        """
        # Create a new NumPy array to store updated centroids
        new_mew = np.empty((self.num_clusters, len(self.training_set[0]) - 1))

        for i, (centroid, points) in enumerate(centroids.items()):
            if points:  # Avoid division by zero
                # Convert list of points to a NumPy array
                points_array = np.array(points)

                # Compute the mean of each feature vector across all points assigned to a given centroid
                new_centroid = points_array.mean(axis=0)

                # Update the new_mew array with the new centroid values
                new_mew[i] = new_centroid
            else:
                # If no points were assigned to the centroid, keep it the same
                new_mew[i] = np.array(centroid)

        return new_mew # Return new centroid locations

    def assign_centroid_values(self):
        """
        Function used to assign centroid values (class/target value) after clustering

        :return: A new set that represents the cultured data set
        """
        centroid_set = []

        # Iterate over the number of clusters
        for i in range(self.num_clusters):
            # Add a zero onto the end of each centroid vector to create room for a class or target value.
            # Append this extended centroid vector to our centroid set.
            centroid_set.append(np.append(self.centroids_locations[i], 0).tolist())

        # Call KNN with the centroid set as the test set to get predicted values. (this will be used as the accepted values for our centroids)
        knn = KNN(self.training_set, centroid_set, self.is_class)
        values = knn.classify_all(1, 1) # Let K = 1 and Sigma = 1 for assigning values to

        num_features = self.num_features
        for i in range(self.num_clusters):
            centroid_set[i][num_features] = values[i] # Overwrite the 0 in the last column of each centroid with the new values derived from KNN

        self.centroid_set = centroid_set

    def print_point_to_cluster(self, point, distances):
        """
        Function to print the assignment of a point to a cluster
        :param point: The data point being assigned
        :param distances: Dictionary of centroid-distance pairs
        :return: None
        """
        print(f"Assigning point {point}\n")
        for centroid, distance in distances.items():
            print(f"  Centroid location: {centroid} \n\t Distance from point: {distance}")

        closest_centroid = argmin(distances)
        print(f"\nClosest Centroid: {closest_centroid}\nAssigned point to cluster\n")











