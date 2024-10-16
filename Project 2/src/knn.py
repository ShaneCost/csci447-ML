__author__ = "<Hayden Perusich>"

import numpy as np
from collections import Counter

class KNN(object):
    def __init__(self, training_set, test_set, is_classification=True):
        # Convert training and testing sets to NumPy arrays for easier manipulation
        self.training_data = np.array(training_set)
        self.testing_data = np.array(test_set)
        # Determine if the task is classification or regression
        self.is_classification = is_classification

    # Classify a single test point using k-nearest neighbors
    def classify(self, test_point, k, sigma=1, print_distances=False, print_rbf=False):
        # List to hold distances from the test point to each training point
        distances = []
        for data_point in self.training_data:
            # Calculate distance and append to distances list
            distance = self.get_distance(test_point[:-1], data_point[:-1])
            distances.append((distance, data_point[-1]))

        # Sort distances based on the distance value
        distances = sorted(distances)
        # Use the voting mechanism to make a prediction
        prediction = self.vote(distances, k, sigma)

        if print_distances:
            # Print the distances of the k nearest neighbors if requested
            print(distances[:k])
        if print_rbf:
            self.print_rbf(distances[:k], sigma)

        return prediction

    # Classify all test points
    def classify_all(self, k, sigma=1):
        predictions = []
        for test_point in self.testing_data:
            # Classify each test point
            prediction = self.classify(test_point, k, sigma)
            predictions.append(prediction)

        return predictions

    # Voting mechanism for classification or averaging for regression
    def vote(self, distances, k, sigma):
        # Get the k nearest neighbors
        distances = distances[:k]
        if self.is_classification:
            # For classification, find the most common class among the neighbors
            classes = [t[1] for t in distances]
            prediction = Counter(classes).most_common(1)[0][0]
        else:
            # For regression, use RBF kernel for weighted average
            neighbor_targets = np.array([t[1] for t in distances], dtype=np.float64)
            distances = np.array([t[0] for t in distances], dtype=np.float64)
            kernel_values = self.rbf_kernel(distances, sigma)
            weights = kernel_values / np.sum(kernel_values)
            # Compute the weighted sum of target values
            prediction = np.dot(weights, neighbor_targets)

        return prediction

    # Radial basis function (RBF) kernel
    def rbf_kernel(self, distances, sigma):
        # Compute the RBF kernel values based on distances
        return np.exp(-sigma * distances ** 2)

    def print_rbf(self, distances, sigma):
        neighbor_targets = np.array([t[1] for t in distances], dtype=np.float64)
        distances = np.array([t[0] for t in distances], dtype=np.float64)
        kernel_values = np.exp(-sigma * distances ** 2)
        weights = kernel_values / np.sum(kernel_values)
        prediction = np.dot(weights, neighbor_targets)
        print("kernel values = (",-sigma, "*", distances,")^2")
        print("weights = ", kernel_values, "/", np.sum(kernel_values))
        print("prediction = ", weights, "*", neighbor_targets)
        print(prediction)

    # Get the actual class or target value for a specific point
    def get_actual(self, point):
        actual = point[-1]
        return float(actual) if not self.is_classification else actual

    # Retrieve actual values for all test points
    def get_actual_all(self):
        actual = [point[-1] for point in self.testing_data]
        return actual

    # Calculate Minkowski distance between two points
    def get_distance(self, x, y, p=2):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        # Calculate the distance based on the given p-value
        distance = np.sum(np.abs(x - y) ** p) ** (1 / p)
        return distance

