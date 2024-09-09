__author__ = "<Hayden Perusich>"

class TenFold:

    def __init__(self):
        """
        Class to store data folds to perform 10-fold cross validation
            Also stores other important information about the data set
        """
        self.data = [] # Stores unsegmented data

        self.num_classes = 0
        self.classes = []
        self.num_features = 0
        self.num_entries = 0

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

    def load(self, path):
        """
        Reads values from pre-processed .data file, collects information
        from file, and groups it into 10 groups (as evenly as it can)

        :param path: String path to the data set
        :return: None
        """
        with open(path, "r") as file:
            num_lines = 0 # Number of data points in file
            num_features = 0 # Number of features per data point
            classes = [] # A list of all classes found in the data set

            for line in file:
                num_lines += 1 # Increment count of data points
                split = line.strip('\n').split(",")
                if len(split) > 1: # Exclude null lines tailing the file
                    self.data.append(split) # Add the line to the 'raw' data attribute
                    num_features = len(split) - 1 # Number of features = length - 1 to account of the class included on the line
                    class_name = split[num_features] # Class name is found at the last index of the line
                    if not class_name in classes: # Append class name to list of all possible classes if not already present
                        classes.append(class_name)
            num_classes = len(classes)

        # Save important data attributes to the object
        self.num_classes = num_classes
        self.classes = classes
        self.num_features = num_features
        self.num_entries = num_lines

        file.close() # Close file

        # Calculate number of entries per fold
        num_entries_per_fold = num_lines // 10
        extra_entries = num_lines % 10  # Remainder = extra entries to be spread throughout data sets

        # Split the data into 10 folds
        folds = []
        start_idx = 0

        for i in range(10):
            end_idx = start_idx + num_entries_per_fold # Variable to track the last index we used to access our data
            if i < extra_entries:  # Add one extra entry to the first 'extra_entries' folds
                end_idx += 1
            folds.append(self.data[start_idx:end_idx])
            start_idx = end_idx # Reset where we start indexing into data

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

def main():
    """
    Demonstration of how to instantiate and use class

    :return: None
    """
    breast_cancer = TenFold()
    breast_cancer.load("../data/processed_data/breast-cancer-wisconsin_processed.data")

    glass = TenFold()
    glass.load("../data/processed_data/glass_processed.data")

    soybean = TenFold()
    soybean.load("../data/processed_data/soybean-small_processed.data")

    iris = TenFold()
    iris.load("../data/processed_data/iris_processed.data")

    votes = TenFold()
    votes.load("../data/processed_data/house-votes-84_processed.data")

    breast_cancer_noisy = TenFold()
    breast_cancer_noisy.load("../data/noisy_data/breast-cancer-wisconsin_noisy.data")

    glass_noisy = TenFold()
    glass_noisy.load("../data/noisy_data/glass_noisy.data")

    soybean_noisy = TenFold()
    soybean_noisy.load("../data/noisy_data/soybean-small_noisy.data")

    iris_noisy = TenFold()
    iris_noisy.load("../data/noisy_data/iris_noisy.data")

    votes_noisy = TenFold()
    votes_noisy.load("../data/noisy_data/house-votes-84_noisy.data")

# main()