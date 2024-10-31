__author__ = "<Shane Costello>"

import numpy as np
from confusion_matrix import *

class Loss:
    def __init__(self, actual, predicted, is_class=True):
        """
        Class to implement the loss functions.
            Classification: Confusion Matrix
            Regression: Mean Squared Error

        :param actual: array of actual values
        :param predicted: array of values predicted from classification/regression
        :param is_class: boolean indicating whether it is a classification or regression problem
        """
        self.actual = actual
        self.predicted = predicted

        # If it is a classification problem
        #   Generate confusion matrix
        #   Use the macro-F1 score as a. standard performance metric
        if is_class:
            self.is_class = True
            self.confusion_matrix = ConfusionMatrix(self.actual, self.predicted)
            self.results = self.confusion_matrix.test_score
       # If it is a regression problem
        #   Calculate the mean squared error
        #   Multiply by negative one to make metric work for tuning
        else:
            self.is_class = False
            self.results = round(-1 * self.mean_squared_error(), 3)

    def mean_squared_error(self):
        """
        Method to calculate the mean squared error.
        Classify as error if it is outside the threshold range.
        :return: mean squared error
        """
        total_error = 0
        n = len(self.actual)

        # Create NumPy arrays
        actual = np.array(self.actual, dtype=np.float64)
        predicted = np.array(self.predicted, dtype=np.float64)

        # Compare actual and predicted value
        for actual, pred in zip(actual, predicted):
            total_error += abs(actual - pred) ** 2 # Calculate error
        return total_error / n # Divide total error by number of data points
