__author__ = "<Shane Costello>"

import numpy as np

class Hyperparameter:
    def __init__(self, name, min_value, max_value):
        """
        Class used to model Hyperparameters

        :param name: string name of the hyperparameter
        :param min_value: minimum value of the hyperparameter
        :param max_value: maximum value of the hyperparameter
        """
        self.name = name
        self.value = 0
        self.min = min_value
        self.max = max_value
        self.values_performance = {}
        self.is_tuned = False

        self.populate_dictionary()

    def populate_dictionary(self):
        """
        Populates the values_performance dictionary with 100 values between
        min_value and max_value, where each key is one of the values and the
        corresponding value is set to 0 (indicating initial performance).
        """
        # Generate 100 values between min and max
        values = np.linspace(self.min, self.max, 50)

        # Populate dictionary with each value as a key and performance set to 0
        self.values_performance = {value: 0 for value in values}

        # Set self.value to the first key in the dictionary
        self.value = list(self.values_performance.keys())[0]

    def update(self, performance):
        """
        Stores the performance metric of the hyperparameter at its current value then
        updates the value to the next key in the values_performance dictionary.
        If value is the last key, it stays the same (or could loop back to the first key).
        """

        self.values_performance[self.value] = performance

        keys = list(self.values_performance.keys())

        # Find the index of the current value
        current_index = keys.index(self.value)

        # Update to the next value if it's not the last one
        if current_index < len(keys) - 1:
            self.value = keys[current_index + 1]
        else:
            self.find_optimal_value()

    def find_optimal_value(self):
            """
            Finds the hyperparameter value with the best performance metric
            """

            self.value = max(self.values_performance, key=self.values_performance.get)
            self.is_tuned = True


