import numpy as np
from collections import Counter

class KNN(object):
    def __init__(self, training_data, testing_data, is_classification=True):
        self.training_data = np.array(training_data)
        self.testing_data = np.array(testing_data)
        # if False assumes regression
        self.is_classification = is_classification

        # k is number of neighbors used in voting or average and p is the p variable in the distance formula

    def classify(self, test_point, k, p):

        # get distance of a point
        distances = []
        for data_point in self.training_data:
            distance = self.get_distance(test_point[:-1], data_point[:-1], p)
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
        distances = distances[0:k]
        if (self.is_classification):
            classes = [t[1] for t in distances]
            prediction = Counter(classes).most_common(1)[0][0]
        else:
            # impliment the kernal function
            target_values = [t[1] for t in distances]
            prediction = sum(target_values) / len(target_values)

        # prediction for class or target_value
        return prediction

    def get_actual(self, point):

        # returns the actual classes/target_values for a point
        actual = point[-1]
        return actual

    def get_actual_all(self):

        actual = []
        for point in self.testing_data:
            actual.append(point[-1])
        return actual

    def get_distance(self, x, y, p):

        # Minkowski distance
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        distance = np.sum(np.abs(x - y) ** p) ** (1 / p)

        return distance

from data import Data
def main():

    path = "Project 2\data\soybean-small.data"
    data = Data(path, "class")

    training_set = data.get_training_set(1)
    test_set = data.get_test_set(1)

    knn = KNN(training_set, test_set)
    predications = knn.classify_all(1, 2)
    actual = knn.get_actual_all()

    print(actual)
    print(predications)

main()
# main()

#
# def rbf_kernel(distances, gamma=1.0):
#     """Computes the RBF kernel values from distances."""
#     return np.exp(-gamma * distances ** 2)
#
#
# def predict_knn_rbf_from_neighbors(neighbors, query_point, gamma=1.0):
#     """
#     Predict the target value using kernel-weighted k-NN regression, given k neighbors.
#
#     Parameters:
#     - neighbors: A numpy array of shape (k, n_features + 1) where the last column is the target value.
#     - query_point: The query point (1D numpy array of shape (n_features,))
#     - gamma: RBF kernel parameter (default = 1.0)
#
#     Returns:
#     - y_pred: Predicted target value (float)
#     """
#     # Separate features and target values from the neighbors array
#     neighbor_features = neighbors[:, :-1]  # All columns except the last one
#     neighbor_targets = neighbors[:, -1]  # Last column contains the target values
#
#     # Compute the Euclidean distances between the query point and the neighbors
#     distances = np.linalg.norm(neighbor_features - query_point, axis=1)
#
#     # Compute RBF kernel values based on the distances
#     kernel_values = rbf_kernel(distances, gamma)
#
#     # Normalize the kernel values to get weights
#     weights = kernel_values / np.sum(kernel_values)
#
#     # Compute the weighted sum of the target values
#     y_pred = np.dot(weights, neighbor_targets)
#
#     return y_pred