import numpy as np
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
                    class_name = split[num_features]
                    if not class_name in classes:
                        classes.append(class_name)
                    self.class_map.append(class_name)
            num_classes = len(classes)

            print("num classes: ", num_classes)
            print("classes: " , classes)
            print("num features: ", num_features)
            print("num entries: ", num_lines)

        file.close()

        with open(path, "r") as file:
            averages = [0] * num_features
            data = [[0] * num_features for _ in range(num_lines)]
            line_count = 0
            for line in file:
                split = line.strip('\n').split(",")
                if len(split) > 1:
                    for i in range(num_features):
                        if not split[i] == '?':
                            averages[i] += float(split[i])
                            data[line_count][i] += float(split[i])

                line_count += 1


            for i in range(num_features):
                averages[i] /= num_lines
                averages[i] = round(averages[i], 2)

            self.num_classes = num_classes
            self.classes = classes
            self.num_features = num_features
            self.num_entries = num_lines
            self.raw_data = data

        file.close()

    def get_name(self, path):
        split = path.split('/')
        name = split[len(split) - 1]
        name_split = name.split('.')
        self.name = name_split[0]

        print("name: ", self.name)

    def bin(self):
        print("binning data...")
        # Step 1: Create an array of arrays for each feature
        feature_arrays = [[] for _ in range(self.num_features)]

        # Populate feature arrays with the respective column data
        for entry in self.raw_data:
            for i in range(len(entry)):
                feature_arrays[i].append(entry[i])

        # Step 2: Calculate the quartiles for each feature
        quartiles = []
        for feature in feature_arrays:
            q1 = np.percentile(feature, 25)
            q2 = np.percentile(feature, 50)  # Median
            q3 = np.percentile(feature, 75)
            q4 = np.percentile(feature, 100)  # Maximum (100th percentile)

            quartiles.append((q1, q2, q3, q4))

        # Step 3: Assign each entry to a quartile
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
            binned_entry.append(self.class_map[entry_num])
            binned_data.append(binned_entry)
            entry_num += 1

        self.binned_data = binned_data

    def shuffle(self):
        print("shuffling data entries...")
        # Make a copy of the original list to avoid modifying it
        shuffled_array = self.binned_data[:]

        # Shuffle the list in place
        random.shuffle(shuffled_array)

        self.shuffled_data = shuffled_array

    def write_data_to_file(self):
        print("writing preprocessed data to file...")
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
        print("adding noise...")
        noise_array = self.shuffled_data[:]
        amount_of_noise = math.ceil(.1 * self.num_features)

        print("num features to be shuffled: ", amount_of_noise)

        already_done_features = []

        for i in range(amount_of_noise):
            feature_num = random.randint(0, self.num_features - 1)

            if feature_num not in already_done_features:
                already_done_features.append(feature_num)

                print("feature to be shuffled:", feature_num)

                num_shuffles = 0
                already_shuffled_data = []
                while num_shuffles <= self.num_entries:
                    entry_1_index = random.randint(0, self.num_entries - 1)
                    entry_2_index = random.randint(0, self.num_entries - 1)

                    if (entry_1_index not in already_shuffled_data and
                            entry_2_index not in already_shuffled_data):
                        entry_1 = noise_array[entry_1_index][feature_num]
                        entry_2 = noise_array[entry_2_index][feature_num]

                        noise_array[entry_1_index][feature_num] = entry_2
                        noise_array[entry_2_index][feature_num] = entry_1
                        already_shuffled_data.append(entry_1_index)
                        already_shuffled_data.append(entry_2_index)

                        num_shuffles += 2
            else:
                i -= 1
                print("feature already shuffled")

        print(noise_array)
        print(self.shuffled_data)

        print("writing noisy data to file...")

        filename = self.name + "_noisy.data"

        current_dir = os.path.dirname(__file__)  # Gets the directory of the current file (src folder)
        data_folder = os.path.join(current_dir, '..', 'data')  # Navigate up one level and into the data folder
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, filename)

        with open(file_path, 'w') as file:
            for entry in noise_array:
                # Convert each sub-array to a string with values separated by spaces
                line = ','.join(map(str, entry))
                # Write the line to the file
                file.write(line + '\n')

def main():
    breast_cancer = Data()
    breast_cancer.process_file("../data/breast-cancer-wisconsin.data")
    breast_cancer.bin()
    breast_cancer.shuffle()
    breast_cancer.write_data_to_file()
    breast_cancer.add_noise()
    print('\n')

    glass = Data()
    glass.process_file("../data/glass.data")
    glass.bin()
    glass.shuffle()
    glass.write_data_to_file()
    glass.add_noise()
    print('\n')

    soybean = Data()
    soybean.process_file("../data/soybean-small.data")
    soybean.bin()
    soybean.shuffle()
    soybean.write_data_to_file()
    soybean.add_noise()
    print('\n')

    iris = Data()
    iris.process_file("../data/iris.data")
    iris.bin()
    iris.shuffle()
    iris.write_data_to_file()
    iris.add_noise()
    print('\n')

# main()

