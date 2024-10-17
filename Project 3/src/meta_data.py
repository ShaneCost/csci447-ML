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
        self.feature_vectors = np.array(data_set)[:, :-1]
        self.feature_vectors.astype(np.float64)

        if is_string(np.array(data_set)[:, -1]):
            self.target_vector = np.array(data_set)[:, -1]
            self.target_vector.astype(object)
        else:
            self.target_vector = np.array(data_set)[:, -1]
            self.target_vector.astype(np.float64)
