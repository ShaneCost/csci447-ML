__author__ = "<Shane Costello>"

import numpy as np
import pandas as pd
import math
import random
from hyperparameter import *


def is_cat(col):
    """
    Function to determine if a column is categorical (contains letters)

    :param col: feature vector
    :return: True if there are alphabetic values, otherwise False
    """
    return any(c.isalpha() for c in ''.join(col.astype(str)))

def normalize(column):
    """
    Function to 0-1 normalize a feature vector
    :param column: feature vector
    :return: 0-1 normalized feature vector
    """
    # Convert raw_data to a NumPy array
    feature_vector = np.array(column, dtype=float)

    min_v = np.min(feature_vector) # Get minimum value in the feature vector
    max_v = np.max(feature_vector) # Get maximum value in the feature vector
    # Handle case where max is equal to min to avoid division by zero
    if max_v == min_v:
        return np.zeros_like(feature_vector)  # Return a zero vector if all values are the same

    normalized_vector = (feature_vector - min_v) / (max_v - min_v)

    return normalized_vector

class Data:
    def __init__(self, path, class_or_regress):
        """
        Class used to read data in from file, performing processing, and create folds

        :param path: string path to a data file
        :param class_or_regress: string denoting whether it is a classification or regression problem
        """
        # Attributes derived from data
        self.path = path
        self.name = ""
        self.raw_data = []
        self.num_features = 0
        self.num_entries = 0

        if class_or_regress == 'class':
            self.is_class = True
        else:
            self.is_class = False

        self.num_classes = 0
        self.classes = []

        # Folds + tuning set
        self.tuning = []
        self.fold_1 = []
        self.fold_2 = []
        self.fold_3 = []
        self.fold_4 = []
        self.fold_5 = []
        self.fold_6 = []
        self.fold_7 = []
        self.fold_8 = []
        self.fold_9 = []
        self.fold_10 = []

        # Dictionary used to store hyperparameters
        self.hyperparameters = {}

        self.load()

    def load(self):
        """
        Function called in instantiation to process file
        :return: None
        """
        self.get_name()
        self.process_file()
        self.one_hot_encode_and_normalize()
        self.set_to_float()

        # Check if folds are evenly distributed
        good_folds = False
        while not good_folds:
            if self.is_class:
                self.class_stratified_ten_fold()
            else:
                self.regress_stratified_ten_fold()

            good_folds = self.check_folds()

        # Create instances of Hyperparameter class
        k = Hyperparameter('k', 1, len(self.tuning), 1)
        self.hyperparameters['k'] = k

        if not self.is_class: # If it is a regression problem
            epsilon_value, epsilon_max, epsilon_step = self.generate_starting_epsilon()
            sigma_value, sigma_max, sigma_step = self.generate_starting_sigma()

            epsilon = Hyperparameter('epsilon', epsilon_value, epsilon_max, epsilon_step)
            sigma = Hyperparameter('sigma', sigma_value, sigma_max, sigma_step)
            self.hyperparameters['epsilon'] = epsilon
            self.hyperparameters['sigma'] = sigma
        else: # If it is a classification problem
            self.hyperparameters['epsilon'] = Hyperparameter('epsilon', 0, 0, 0)
            self.hyperparameters['sigma'] = Hyperparameter('sigma', 0, 0, 0)

    def get_name(self):
        """
        Parse the string 'path' to retrieve name of data set.
        Set attribute of class.

        :return: None
        """
        split = self.path.split('/')
        name = split[len(split) - 1]
        name_split = name.split('.')
        self.name = name_split[0]

    def process_file(self):
        """
        Reads values from processed data file. Fills raw_data attribute
        and collects important information about the data set
        (see attributes above)

        :return: None
        """
        # Open file to collect preliminary information
        with open(self.path, "r") as file:
            num_lines = 0 # Number of data points in file
            num_features = 0 # Number of features per data point
            for line in file:
                num_lines += 1 # Increment count of data points
                split = line.strip('\n').split(",")
                if len(split) > 1: # Exclude null lines tailing the file
                    num_features = len(split) - 1 # Number of features = length - 1 to account of the class included on the line

        file.close()

        self.num_features = num_features
        self.num_entries = num_lines

        # Reopen file to file raw_data attribute
        with open(self.path, "r") as file:
            line_count = 0
            data = np.empty((num_lines, num_features + 1), dtype=object)
            for line in file:
                split = line.strip('\n').split(",")
                if len(split) > 1: # Exclude null lines tailing the file
                    for i in range(num_features + 1):
                        data[line_count, i] = split[i]
                line_count += 1

        file.close()

        self.raw_data = data

        # If it is a classification problem, collected data about classes
        if self.is_class:
            with open(self.path, "r") as file:
                classes = [] # A list of all classes found in the data set
                for line in file:
                    split = line.strip('\n').split(",")
                    if len(split) > 1:  # Exclude null lines tailing the file
                        class_name = split[num_features] # Class name is found at the last index of the line
                        if not class_name in classes: # Append class name to list of all possible classes if not already present
                            classes.append(class_name)
                num_classes = len(classes)

            file.close()

            self.num_classes = num_classes
            self.classes = classes

    def one_hot_encode_and_normalize(self):
        """
        Function to apply one hot encoding to all categorical feature vectors and
        normalize the feature vectors.

        :return: None
        """
        df = pd.DataFrame(self.raw_data) # Convert self.raw_data into a pandas DataFrame

        target_column = df.iloc[:, -1]  # Get the last column (target)
        feature_columns = df.iloc[:, :-1]  # Get all columns except the last one

        cat_mask = feature_columns.apply(is_cat, axis=0) # Apply the function to each column
        cat_cols = feature_columns.columns[cat_mask] # Get the names of categorical columns using the mask
        df_encoded = pd.get_dummies(feature_columns, columns=cat_cols, drop_first=False) # Apply one-hot encoding to the categorical columns
        normalized_features = df_encoded.apply(normalize, axis=0) # Normalize all feature columns

        df_normalized = pd.concat([normalized_features, target_column], axis=1) # Reconstruct the DataFrame with normalized features and the target column

        # Get the new number of features with one-hot encoding
        row, col = df_normalized.shape
        self.num_features = col - 1

        self.raw_data = df_normalized.to_numpy() # Convert from dataframe back to numpy array

    def set_to_float(self):
        """
        Function to convert values from strings to floats
        :return: None
        """
        for row in self.raw_data:
            for i in range(len(row) - 1):
                row[i] = float(row[i])

    def set_tuning(self):
        """
        Function to remove 10% of data at random and set it to be our tuning data set

        :return: Modified data set (raw data without the values removed for tuning)
        """
        for_ten_fold = self.raw_data.copy().tolist() # Create a copy of the raw data to modify
        tuning_size = math.floor(len(self.raw_data) * .1) # Generate size of tuning set (10% of total raw data)

        tuning = []
        removed = []
        removed_num = 0

        while removed_num < tuning_size: # While the tuning set is not full
            index = random.randint(0, len(for_ten_fold) - 1) # Randomly generate index within the data set
            if index not in removed: # If the index has not already been removed
                removed.append(index)
                tuning.append(for_ten_fold.pop(index)) # Append the value at that index to the tuning set, pop it off raw data
                removed_num += 1

        self.tuning = tuning # Set attribute
        return for_ten_fold # Return the modified data set

    def class_stratified_ten_fold(self):
        """
        Function to create 10 stratified folds for a classification problem

        :return: None
        """
        data = self.set_tuning() # Derive edited data set

        # Initialize the folds
        folds = [[] for _ in range(10)]

        # Create a mapping of classes to their respective instances
        class_set = {}
        for class_name in self.classes:
            class_set[class_name] = []
        for row in data:
            class_set[row[-1]].append(row)

        # Distribute instances into folds
        for class_name, instances in class_set.items():
            np.random.shuffle(instances)  # Shuffle instances of the current class

            # Total instances of the current class
            num_instances = len(instances)
            instances_per_fold = num_instances // 10
            extra_instances = num_instances % 10

            index = 0
            for j in range(10):
                # Number of instances to add to this fold
                instances_to_add = instances_per_fold + (1 if j < extra_instances else 0)
                for _ in range(instances_to_add):
                    if instances:  # Ensure there are still instances to take
                        folds[j].append(instances.pop())

        # Shuffle each fold to ensure randomness
        for fold in folds:
            np.random.shuffle(fold)

        # Assign folds to class attributes
        self.fold_1 = folds[0]
        self.fold_2 = folds[1]
        self.fold_3 = folds[2]
        self.fold_4 = folds[3]
        self.fold_5 = folds[4]
        self.fold_6 = folds[5]
        self.fold_7 = folds[6]
        self.fold_8 = folds[7]
        self.fold_9 = folds[8]
        self.fold_10 = folds[9]

    def regress_stratified_ten_fold(self):
        """
        Function to create 10 stratified folds for a regression problem

        :return: None
        """
        data = self.set_tuning() # Derive edited data set

        data.sort(key=lambda x: x[-1]) # Sort values based on target value

        # Split the data into groups of ten
        groups_of_10 = []
        start_idx = 0
        end_idx = 0
        while end_idx < len(data):
            end_idx = start_idx + 10

            if end_idx > len(data) - 1:
                end_idx = len(data) - 1
                groups_of_10.append(data[start_idx:end_idx])
                end_idx = len(data)

            groups_of_10.append(data[start_idx:end_idx])
            start_idx = end_idx

        # Pull values from groups of ten to create stratified folds
        folds = [[] for _ in range(10)]
        for i in range(10):
            for group in groups_of_10:
                max_value = len(group)
                if max_value < i + 1:
                    pass
                else:
                    folds[i].append(group[i])

        # Assign folds to class attributes
        self.fold_1 = folds[0]
        self.fold_2 = folds[1]
        self.fold_3 = folds[2]
        self.fold_4 = folds[3]
        self.fold_5 = folds[4]
        self.fold_6 = folds[5]
        self.fold_7 = folds[6]
        self.fold_8 = folds[7]
        self.fold_9 = folds[8]
        self.fold_10 = folds[9]

    def check_folds(self):
        """
        Function to check if the size variance among folds is 20% or less

        :return: True if less than or equal to 20% variance, False otherwise
        """
        folds = [0 for _ in range(10)]
        folds[0] = len(self.fold_1)
        folds[1] = len(self.fold_2)
        folds[2] = len(self.fold_3)
        folds[3] = len(self.fold_4)
        folds[4] = len(self.fold_5)
        folds[5] = len(self.fold_6)
        folds[6] = len(self.fold_7)
        folds[7] = len(self.fold_8)
        folds[8] = len(self.fold_9)
        folds[9] = len(self.fold_10)

        if max(folds) * .80 <= min(folds):
            return True
        else:
            return False

    def get_training_set(self, test_set_num):
        """
        Creates a training data set based on which fold is your current test set

        :param test_set_num: number value of test set to be excluded from training
        :return: 2D array containing all folds in the training data set
        """
        training_data = []

        if test_set_num != 1:
            training_data.extend(self.fold_1)
        if test_set_num != 2:
            training_data.extend(self.fold_2)
        if test_set_num != 3:
            training_data.extend(self.fold_3)
        if test_set_num != 4:
            training_data.extend(self.fold_4)
        if test_set_num != 5:
            training_data.extend(self.fold_5)
        if test_set_num != 6:
            training_data.extend(self.fold_6)
        if test_set_num != 7:
            training_data.extend(self.fold_7)
        if test_set_num != 8:
            training_data.extend(self.fold_8)
        if test_set_num != 9:
            training_data.extend(self.fold_9)
        if test_set_num != 10:
            training_data.extend(self.fold_10)

        return training_data

    def get_test_set(self, test_set_num):
        """
        Provides the fold equivalent to the current test set

        :param test_set_num: number value of test set to be returned
        :return: 2D array containing the fold of the test data set
        """
        if test_set_num == 1:
            return self.fold_1
        elif test_set_num == 2:
            return self.fold_2
        elif test_set_num == 3:
            return self.fold_3
        elif test_set_num == 4:
            return self.fold_4
        elif test_set_num == 5:
            return self.fold_5
        elif test_set_num == 6:
            return self.fold_6
        elif test_set_num == 7:
            return self.fold_7
        elif test_set_num == 8:
            return self.fold_8
        elif test_set_num == 9:
            return self.fold_9
        elif test_set_num == 10:
            return self.fold_10

    def generate_starting_epsilon(self):
        """
        Function to generate data pertaining to the hyperparameter epsilon

        :return: Starting value, max value, step value
        """
        target_values = [float(row[-1]) for row in self.raw_data] # Pull of the target value vector
        epsilon_start = 0.01 * (max(target_values) - min(target_values)) # Starting = 1% of variance in target values
        epsilon_step = 0.001 * (max(target_values) - min(target_values)) # Step = 0.1% of variance in target values
        epsilon_max = 0.20 * (max(target_values) - min(target_values)) # Max = 20% of variance in target values

        return round(epsilon_start, 3), round(epsilon_max, 3), round(epsilon_step, 3)

    def generate_starting_sigma(self):
        """
        Function to generate data pertaining to the hyperparameter sigma

        :return: Starting value, max value, step value
        """
        data = np.array(self.raw_data, dtype=float) # Create NumPy array
        ranges = np.ptp(data[:, :-1], axis=0) # Pull all feature vectors together and take the average range
        sigma_start = 0.1 * np.mean(ranges) # Start = 10% of the average feature range
        sigma_step = 0.01 * np.mean(ranges) # Step = 1% of the average feature range
        max_value = 1 # Max = 1

        return round(sigma_start, 3), round(max_value, 3), round(sigma_step, 3)




