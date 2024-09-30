import numpy as np
import pandas as pd
import math
import random
from hyperparameter import *

# Function to determine if a column is categorical (contains letters)
def is_cat(col):
    return any(c.isalpha() for c in ''.join(col.astype(str)))

# Function to 0-1 normalize a feature vector
def normalize(column):
    # Convert raw_data to a NumPy array for easier manipulation
    feature_vector = np.array(column, dtype=float)

    min_v = np.min(feature_vector)
    max_v = np.max(feature_vector)
    # Handle case where max_v is equal to min_v to avoid division by zero
    if max_v == min_v:
        return np.zeros_like(feature_vector)  # Return a zero vector if all values are the same

    normalized_vector = (feature_vector - min_v) / (max_v - min_v)

    return normalized_vector

class Data:
    def __init__(self, path, class_or_regress):
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

        self.hyperparameters = {}

        self.load()

    def load(self):
        self.get_name()
        self.process_file()
        self.one_hot_encode_and_normalize()
        self.set_to_float()

        good_folds = False
        while not good_folds:
            if self.is_class:
                self.class_stratified_ten_fold()
            else:
                self.regress_stratified_ten_fold()

            good_folds = self.check_folds()

        k = Hyperparameter('k', 1, 1)
        p = Hyperparameter('p', 1, 1)
        self.hyperparameters['k'] = k
        self.hyperparameters['p'] = p

        if not self.is_class:
            epsilon_value, epsilon_step = self.generate_starting_epsilon()
            sigma_value, sigma_step = self.generate_starting_sigma()

            epsilon = Hyperparameter('epsilon', epsilon_value, epsilon_step)
            sigma = Hyperparameter('sigma', sigma_value, sigma_step)
            self.hyperparameters['epsilon'] = epsilon
            self.hyperparameters['sigma'] = sigma


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
        df = pd.DataFrame(self.raw_data) # Convert self.raw_data into a pandas DataFrame

        # Separate the target column
        target_column = df.iloc[:, -1]  # Get the last column (target)
        feature_columns = df.iloc[:, :-1]  # Get all columns except the last one

        cat_mask = feature_columns.apply(is_cat, axis=0) # Apply the function to each column and get a mask
        cat_cols = feature_columns.columns[cat_mask] # Get the names of categorical columns using the mask
        df_encoded = pd.get_dummies(feature_columns, columns=cat_cols, drop_first=False) # Apply one-hot encoding to the categorical columns
        # Normalize all feature columns
        normalized_features = df_encoded.apply(normalize, axis=0)

        # Reconstruct the DataFrame with normalized features and the target column
        df_normalized = pd.concat([normalized_features, target_column], axis=1)

        row, col = df_normalized.shape
        self.num_features = col - 1

        self.raw_data = df_normalized.to_numpy() # Convert from dataframe back to numpy array

    def set_to_float(self):
        for row in self.raw_data:
            for i in range(len(row) - 1):
                row[i] = float(row[i])

    def set_tuning(self):
        for_ten_fold = self.raw_data.copy().tolist()
        tuning_size = math.floor(len(self.raw_data) * .1)

        tuning = []
        removed = []
        removed_num = 0

        while removed_num < tuning_size:
            index = random.randint(0, len(for_ten_fold) - 1)
            if index not in removed:
                removed.append(index)
                tuning.append(for_ten_fold.pop(index))
                removed_num += 1

        self.tuning = tuning
        return for_ten_fold

    def class_stratified_ten_fold(self):
        data = self.set_tuning()

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
        data = self.set_tuning()

        data.sort(key=lambda x: x[-1])

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
        target_values = [float(row[-1]) for row in self.raw_data]
        epsilon_start = 0.05 * (max(target_values) - min(target_values))
        epsilon_step = 0.1 * (max(target_values) - min(target_values))

        return epsilon_start, epsilon_step

    def generate_starting_sigma(self):
        data = np.array(self.raw_data, dtype=float)
        ranges = np.ptp(data[:, :-1], axis=0)
        sigma = 0.1 * np.mean(ranges) # Set sigma as a fraction (10%) of the average feature range
        return sigma, sigma




