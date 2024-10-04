import numpy as np
from collections import Counter

class KNN(object):
    def __init__(self, training_set, test_set, is_classification=True):

        self.training_data = np.array(training_set)
        self.testing_data = np.array(test_set)
        # if False assumes regression
        self.is_classification = is_classification

    # k is number of neighbors used in voting or average and p is the p variable in the distance formula
    def classify(self, test_point, k, p):

        # get distance of a point
        distances = []
        for data_point in self.training_data:
            distance = self.get_distance(test_point[:-1], data_point[:-1])
            distances.append((distance, data_point[-1]))

        # containes the list of distances and each associated class
        distances = sorted(distances)
        predication = self.vote(distances, k)

        return predication

    def classify_all(self, k, p):

        predictions = []
        for test_point in self.testing_data:
            prediction = self.classify(test_point, k, p)  # Corrected here
            predictions.append(prediction)

        return predictions


    def vote(self, distances, k):
        # voting/average (depending on is_classification)
        distances = distances[:k]  # Get the k nearest distances
        if self.is_classification:
            classes = [t[1] for t in distances]
            prediction = Counter(classes).most_common(1)[0][0]
        else:
            # Implement the kernel function
            neighbor_targets = np.array([t[1] for t in distances], dtype=np.float64)
            distances = np.array([t[0] for t in distances], dtype=np.float64)
            kernel_values = self.rbf_kernel(distances)
            weights = kernel_values / np.sum(kernel_values)
            # Compute the weighted sum of the target values
            prediction = np.dot(weights, neighbor_targets)  # Now shapes will align

        # Return prediction for class or target_value
        return prediction

    
    def rbf_kernel(self, distances, sigma=1.0):
        # Computes the RBF kernel values from distances
        return np.exp(-sigma * distances ** 2)

    def get_actual(self, point):

        # returns the actual classes/target_values for a point
        actual = point[-1]
        return actual

    def get_actual_all(self):

        actual = []
        for point in self.testing_data:
            actual.append(point[-1])
        return actual

    def get_distance(self, x, y, p=2):

        # Minkowski distance
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        distance = np.sum(np.abs(x - y) ** p) ** (1 / p)

        return distance

from data import Data
def main():

    path = "Project 2\data\machine.data"
    data = Data(path, "regress")

    training_set = data.get_training_set(1)
    test_set = data.get_test_set(1)

    knn = KNN(training_set, test_set, is_classification=False)
    predications = knn.classify_all(1, 2)
    actual = knn.get_actual_all()

    print(actual)
    print(predications)

main()