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
        noise_array = self.shuffled_data[:]
        amount_of_noise = math.ceil(0.1 * self.num_features)

        print("num features to be shuffled: ", amount_of_noise)

        already_done_features = set()

        for _ in range(amount_of_noise):
            feature_num = random.randint(0, self.num_features - 1)

            if feature_num not in already_done_features:
                already_done_features.add(feature_num)
                print("feature to be shuffled:", feature_num)

                indices = list(range(self.num_entries))
                random.shuffle(indices)
                for i in range(0, len(indices), 2):
                    if i + 1 < len(indices):
                        entry_1_index = indices[i]
                        entry_2_index = indices[i + 1]

                        entry_1 = noise_array[entry_1_index][feature_num]
                        entry_2 = noise_array[entry_2_index][feature_num]

                        noise_array[entry_1_index][feature_num] = entry_2
                        noise_array[entry_2_index][feature_num] = entry_1
            else:
                print("feature already shuffled")

        self._write_noise_file(noise_array)

    def _write_noise_file(self, noise_array):
        print(noise_array)
        print(self.shuffled_data)
        filename = self.name + "_noisy.data"
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', filename)

        with open(file_path, 'w') as file:
            for entry in noise_array:
                line = ','.join(map(str, entry))
                file.write(line + '\n')

def main():
    votes = Data()
    votes.process_file("../data/house-votes-84.data")
    votes.shuffle()
    votes.write_data_to_file()
    votes.add_noise()

# main()
