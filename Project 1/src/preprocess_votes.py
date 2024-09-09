__author__ = "<Shane Costello>"

import random
import os
import math

class Data:
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

        self.get_name(path)

        with open(path, "r") as file:
            num_lines = 0
            num_features = 0
            classes = []
            for line in file:
                num_lines += 1
                split = line.strip('\n').split(",")
                if len(split) > 1:
                    num_features = len(split) - 1
                    class_name = split[0]
                    if not class_name in classes:
                        classes.append(class_name)
                    self.class_map.append(class_name)

                    new_data = split[1:]
                    new_data.append(class_name)
                    self.raw_data.append(new_data)
            num_classes = len(classes)

            print("num classes: ", num_classes)
            print("classes: " , classes)
            print("num features: ", num_features)
            print("num lines: ", num_lines)

            self.num_classes = num_classes
            self.classes = classes
            self.num_features = num_features
            self.num_entries = num_lines

        file.close()

    def get_name(self, path):
        split = path.split('/')
        name = split[len(split) - 1]
        name_split = name.split('.')
        self.name = name_split[0]

        print("name: ", self.name)

    def shuffle(self):
        # Make a copy of the original list to avoid modifying it
        shuffled_array = self.raw_data[:]

        # Shuffle the list in place
        random.shuffle(shuffled_array)

        self.shuffled_data = shuffled_array


    def write_data_to_file(self):
        filename = self.name + "_processed.data"

        current_dir = os.path.dirname(__file__)  # Gets the directory of the current file (src folder)
        data_folder = os.path.join(current_dir, '..', 'data')  # Navigate up one level and into the data folder
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, filename)

        with open(file_path, 'w') as file:
            for entry in self.shuffled_data:
                # Convert each sub-array to a string with values separated by spaces
                line = ','.join(map(str, entry))
                # Write the line to the file
                file.write(line + '\n')

    def add_noise(self):
        noise_array = self.shuffled_data[:]  # Make a copy of the original list to avoid modifying it

        amount_of_noise = math.ceil(.1 * self.num_entries)  # Calculate the number of features to be shuffled
        print("num features to be shuffled: ", amount_of_noise)

        for i in range(self.num_features):
            num_shuffles = 0
            already_shuffled_data = []
            while num_shuffles <= amount_of_noise:  # Iterate until every datapoint has been shuffled
                # Randomly generate 2 indexes
                entry_1_index = random.randint(0, self.num_entries - 1)
                entry_2_index = random.randint(0, self.num_entries - 1)

                # Ensure indexes have not already been selected
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
    votes = Data()
    votes.process_file("../data/raw_data/house-votes-84.data")
    votes.shuffle()
    votes.write_data_to_file()
    votes.add_noise()

main()
