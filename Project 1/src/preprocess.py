import numpy as np
import random
import os
import math

class Data:
    """
    Class used to pre-process data (imputation, binning, and shuffling)
    and create noisy data set (shuffle 10% of features randomly)
    """
    def __init__(self):
        self.name = ""
        self.num_classes = 0
        self.classes = []
        self.class_map = []
        self.num_features = 0
        self.num_entries = 0
        self.raw_data = []
        self.binned_data = []
        self.shuffled_data= []

    def process_file(self, path):
        """
        Reads values from unprocessed data file. Outputs average per feature
        (for imputation), fills raw_data attribute, and collects important
        information about the data set (see attributes above)

        :param path: String path to the data set
        :return: None
        """
        self.get_name(path) # Retrieve name of data set

        with open(path, "r") as file: # Open file to collect preliminary information
            num_lines = 0 # Number of data points in file
            num_features = 0 # Number of features per data point
            classes = [] # A list of all classes found in the data set

            for line in file:
                num_lines += 1 # Increment count of data points
                split = line.strip('\n').split(",")
                if len(split) > 1: # Exclude null lines tailing the file
                    num_features = len(split) - 1 # Number of features = length - 1 to account of the class included on the line
                    class_name = split[num_features] # Class name is found at the last index of the line
                    if not class_name in classes: # Append class name to list of all possible classes if not already present
                        classes.append(class_name)
                    self.class_map.append(class_name) # class_map is used to match the class to a data point
            num_classes = len(classes)

            # Print to display outputs
            print("num classes: ", num_classes)
            print("classes: " , classes)
            print("num features: ", num_features)
            print("num entries: ", num_lines)

        file.close()

        with open(path, "r") as file: # Reopen file to collect further information
            averages = [0] * num_features # Initialize an array of size num_features
            data = [[0] * num_features for _ in range(num_lines)] # Initialize an 2D-array of size num_features by num_lines
            line_count = 0
            for line in file:
                split = line.strip('\n').split(",")
                if len(split) > 1: # Exclude null lines tailing the fil
                    for i in range(num_features):
                        if not split[i] == '?': # If not missing data point
                            averages[i] += float(split[i]) # Sum the value of all respective features (at the same index it appears in the file)
                            data[line_count][i] += float(split[i]) # Clone data from file to new data object

                line_count += 1

            file.close()

            # Calculate average value of each feature (used for data imputation)
            for i in range(num_features):
                averages[i] /= num_lines
                averages[i] = round(averages[i], 2)

            # Set data attributes
            self.num_classes = num_classes
            self.classes = classes
            self.num_features = num_features
            self.num_entries = num_lines
            self.raw_data = data

    def get_name(self, path):
        """
        Parse the string 'path' to retrieve name of data set.
        Set attribute of class.

        :param path: String path to the data set
        :return: None
        """
        split = path.split('/')
        name = split[len(split) - 1]
        name_split = name.split('.')
        self.name = name_split[0]

        print("name: ", self.name)

    def bin(self):
        """
        Takes raw data set (continuous values) and performs statistical analysis
        to find quartiles for each feature. Iterates over data set and 'bins'
        raw data depending on which quartile range it falls under.

        Creates  a discrete (categorical) data set

        :return: None
        """
        print("binning data...")

        # Create an array of arrays for each feature
        feature_arrays = [[] for _ in range(self.num_features)]

        # Populate feature arrays with the respective column data
        for entry in self.raw_data:
            for i in range(len(entry)):
                feature_arrays[i].append(entry[i])

        # Calculate the quartiles for each feature
        quartiles = []
        for feature in feature_arrays:
            q1 = np.percentile(feature, 25)
            q2 = np.percentile(feature, 50)  # Median
            q3 = np.percentile(feature, 75)
            q4 = np.percentile(feature, 100)  # Maximum (100th percentile)

            quartiles.append((q1, q2, q3, q4))

        # Assign each entry to a quartile
        binned_data = []
        entry_num = 0

        for entry in self.raw_data:
            binned_entry = []
            for i in range(len(entry)):
                raw_value = entry[i]
                quartile = quartiles[i]
                if raw_value <= quartile[0]:
                    new_value = 'q1'
                elif quartile[0] < raw_value <= quartile[1]:
                    new_value = 'q2'
                elif quartile[1] < raw_value <= quartile[2]:
                    new_value = 'q3'
                else:
                    new_value = 'q4'

                binned_entry.append(new_value)
            binned_entry.append(self.class_map[entry_num]) # Reassign class identifier to data point
            binned_data.append(binned_entry)
            entry_num += 1

        self.binned_data = binned_data

    def shuffle(self):
        """
        Takes binned data set and performs a random shuffle

        :return: None
        """
        print("shuffling data entries...")

        shuffled_array = self.binned_data[:] # Make a copy of the original list to avoid modifying it

        random.shuffle(shuffled_array) # Shuffle the list in place

        self.shuffled_data = shuffled_array

    def write_data_to_file(self):
        """
        Takes shuffled, discrete data and writes it to a .data file

        :return: .data file
        """
        print("writing preprocessed data to file...")

        filename = self.name + "_processed.data" # Create string to be used a filename

        current_dir = os.path.dirname(__file__)  # Gets the directory of the current file (src folder)
        data_folder = os.path.join(current_dir, '..', 'data')  # Navigate up one level and into the data folder
        os.makedirs(data_folder, exist_ok=True) # Make sure data folder exists
        file_path = os.path.join(data_folder, filename) # Create path with new filename

        with open(file_path, 'w') as file: # Open file for writing
            for entry in self.shuffled_data:
                line = ','.join(map(str, entry)) # Convert each entry to a string with values separated by commas
                file.write(line + '\n') # Write the line to the file

    def add_noise(self):
        """
        Takes preprocessed data and adds noise to data set.
        Noise is defined as randomly selecting 10% of values for each feature and shuffling them .

        :return: None
        """
        print("adding noise...")

        noise_array = self.shuffled_data[:] # Make a copy of the original list to avoid modifying it

        amount_of_noise = math.ceil(.1 * self.num_entries) # Calculate the number of values that will be shuffled in each feature
        print("num values to be shuffled: ", amount_of_noise)

        for i in range(self.num_features): # iterate through each feature
                num_shuffles = 0
                already_shuffled_data = [] # used to track which rows (per feature) have already been shuffled
                while num_shuffles <= amount_of_noise: # Iterate until 10% of values have been shuffled
                    # Randomly generate 2 row indexes
                    entry_1_index = random.randint(0, self.num_entries - 1)
                    entry_2_index = random.randint(0, self.num_entries - 1)

                    # Ensure rows have not already been selected (for this feature)
                    if (entry_1_index not in already_shuffled_data and
                            entry_2_index not in already_shuffled_data):

                        # Retrieve value of feature at the generate index
                        entry_1 = noise_array[entry_1_index][i]
                        entry_2 = noise_array[entry_2_index][i]

                        # Shuffle values
                        noise_array[entry_1_index][i] = entry_2
                        noise_array[entry_2_index][i] = entry_1
                        already_shuffled_data.append(entry_1_index)
                        already_shuffled_data.append(entry_2_index)

                        # Increment twice for shuffling two data points
                        num_shuffles += 2

        # Write noisy data to a .data file
        print("writing noisy data to file...")

        filename = self.name + "_noisy.data"

        current_dir = os.path.dirname(__file__)
        data_folder = os.path.join(current_dir, '..', 'data')
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, filename)

        with open(file_path, 'w') as file:
            for entry in noise_array:
                line = ','.join(map(str, entry))
                file.write(line + '\n')

def main():
    """
    Demonstration of how to instantiate and use the class

    :return: None
    """
    breast_cancer = Data()
    breast_cancer.process_file("../data/raw_data/breast-cancer-wisconsin.data")
    breast_cancer.bin()
    breast_cancer.shuffle()
    breast_cancer.write_data_to_file()
    breast_cancer.add_noise()
    print('\n')

    glass = Data()
    glass.process_file("../data/raw_data/glass.data")
    glass.bin()
    glass.shuffle()
    glass.write_data_to_file()
    glass.add_noise()
    print('\n')

    soybean = Data()
    soybean.process_file("../data/raw_data/soybean-small.data")
    soybean.bin()
    soybean.shuffle()
    soybean.write_data_to_file()
    soybean.add_noise()
    print('\n')

    iris = Data()
    iris.process_file("../data/raw_data/iris.data")
    iris.bin()
    iris.shuffle()
    iris.write_data_to_file()
    iris.add_noise()
    print('\n')

main()

