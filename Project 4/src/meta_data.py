__author__ = "<Shane Costello>"

import numpy as np

def is_string(col):
    """
    Function to determine if a column is categorical (contains letters)

    :param col: feature vector
    :return: True if there are alphabetic values, otherwise False
    """
    return any(c.isalpha() for c in ''.join(col.astype(str)))

class MetaData:
    def __init__(self, data_set):
        """
        Class used to represent a data subset (i.e. training set, testing set, tuning set).
        Stores feature vectors separate from target vectors for easy data analysis.

        :param data_set: Subset of data
        """

        # Set feature vectors to be of type float
        self.feature_vectors = np.array(data_set)[:, :-1]
        self.feature_vectors.astype(np.float64)

        # Set target vector to either be of type string or float, depending on contents
        if is_string(np.array(data_set)[:, -1]):
            self.target_vector = np.array(data_set)[:, -1]
            self.target_vector.astype(object)
        else:
            self.target_vector = np.array(data_set)[:, -1]
            self.target_vector.astype(np.float64)
