import numpy as np
from confusion_matrix import *

class Loss:
    def __init__(self, actual, predicted, class_or_regress, epsilon):
        self.actual = actual
        self.predicted = predicted
        self.epsilon = epsilon

        if class_or_regress == 'class':
            self.is_class = True
            self.confusion_matrix = ConfusionMatrix(self.actual, self.predicted)
            self.results = self.confusion_matrix.test_score
        else:
            self.is_class = False
            self.results = round(-1 * self.mean_squared_error(), 3)

    def mean_squared_error(self):
        total_error = 0
        n = len(self.actual)

        actual = np.array(self.actual, dtype=np.float64)
        predicted = np.array(self.predicted, dtype=np.float64)

        for actual, pred in zip(actual, predicted):
            error = abs(actual - pred)
            if error > self.epsilon:
                total_error += (actual - pred) ** 2
            # if the error is within epsilon, we do not add to the total error
        return total_error / n
