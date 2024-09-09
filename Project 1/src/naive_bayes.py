__author__ = "<Hayden Perusich>"

class NaiveBayesClassifier:
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data

        self.class_probabilities = {}
        self.probability_table = {}

    def train(self):
        num_data_points = len(self.training_data)

        class_counts = {}
        class_probabilities = {}

        feature_counts = {}
        feature_probabilities = {}

        num_features = 0

        # Generate Probability of Each Class
        for row in self.training_data:
            num_features = len(row) - 1
            class_name = row[-1]

            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        for class_name in class_counts:
            class_probabilities[class_name] = class_counts[class_name]/num_data_points

        self.class_probabilities = class_probabilities

        # get all possible values for
        possible_values = []
        for row in self.training_data:
            for i in range(len(row) - 1):
                value = row[i]
                if value not in possible_values:
                    possible_values.append(value)

        # Initialize counts table with all 1's for all possible values
        for feature in range(num_features):
            for class_name in class_counts:
                for row in self.training_data:
                    if row[-1] == class_name:

                        if class_name not in feature_counts:
                            feature_counts[class_name] = {}
                            feature_probabilities[class_name] = {}

                        if feature not in feature_counts[class_name]:
                            feature_counts[class_name][feature] = {}
                            feature_probabilities[class_name][feature] = {}

                        for feature_value_1 in possible_values:
                            feature_counts[class_name][feature][feature_value_1] = 1
                            feature_probabilities[class_name][feature][feature_value_1] = 0

        for feature in range(num_features):
            for class_name in class_counts:
                for row in self.training_data:
                    if row[-1] == class_name:
                        feature_value = row[feature]
                        feature_counts[class_name][feature][feature_value] += 1

        for class_name in feature_counts:
            for feature in range(num_features):
                counts = {}
                denominator = 0
                for value in possible_values:
                    count = feature_counts[class_name][feature].pop(value)
                    counts[value] = count
                    denominator += count

                for value in counts:
                    feature_probabilities[class_name][feature][value] = counts[value] / denominator

        self.probability_table = feature_probabilities

    def classify(self, row):

        probability_class = {} # Dic where each class will be assigned a probability that the row is that class
        for class_name in self.class_probabilities:
            probability_value = 1
            
            for i, feature in enumerate (row[:-1]): # Get probability of each class dependent on the column and feature and * together
                probability_value *= self.probability_table[class_name][i][feature]

            probability_value *= self.class_probabilities[class_name]
            probability_class[class_name] = probability_value

        return probability_class

    @staticmethod
    def arg_max(dictionary):
        return max(dictionary, key=dictionary.get)
