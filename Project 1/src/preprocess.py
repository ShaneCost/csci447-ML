import numpy as np
import random
import os

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
            print("num lines: ", num_lines)
            print("class map: ", self.class_map)

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

            print(data)

            for i in range(num_features):
                averages[i] /= num_lines
                averages[i] = round(averages[i], 2)

            print("averages: ", averages)

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
        # Step 1: Create an array of arrays for each feature
        feature_arrays = [[] for _ in range(self.num_features)]

        # Populate feature arrays with the respective column data
        for entry in self.raw_data:
            for i in range(len(entry)):
                feature_arrays[i].append(entry[i])

        print("Feature Arrays: ", feature_arrays)

        # Step 2: Calculate the quartiles for each feature
        quartiles = []
        for feature in feature_arrays:
            q1 = np.percentile(feature, 25)
            q2 = np.percentile(feature, 50)  # Median
            q3 = np.percentile(feature, 75)
            q4 = np.percentile(feature, 100)  # Maximum (100th percentile)

            quartiles.append((q1, q2, q3, q4))

        print("Quartiles: ", quartiles)

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

        print("Binned Data:", binned_data)
        self.binned_data = binned_data

    def shuffle(self):
        # Make a copy of the original list to avoid modifying it
        shuffled_array = self.binned_data[:]

        # Shuffle the list in place
        random.shuffle(shuffled_array)

        self.shuffled_data = shuffled_array

        print("Shuffled data: ", self.shuffled_data)

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


breast_cancer = Data()
breast_cancer.process_file("../data/breast-cancer-wisconsin.data")
breast_cancer.bin()
breast_cancer.shuffle()
breast_cancer.write_data_to_file()
print('\n')

glass = Data()
glass.process_file("../data/glass.data")
glass.bin()
glass.shuffle()
glass.write_data_to_file()
print('\n')

soybean = Data()
soybean.process_file("../data/soybean-small.data")
soybean.bin()
soybean.shuffle()
soybean.write_data_to_file()
print('\n')


