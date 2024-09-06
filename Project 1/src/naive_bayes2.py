from ten_fold import TenFold

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

        possible_values = ['q1', 'q2', 'q3', 'q4']

        # Initialize counts table with all 1's for all possible values
        for feature in range(num_features):
            for class_name in class_counts:
                for row in self.training_data:
                    if row[-1] == class_name:
                        feature_value = row[feature]
                        # print(feature, ": ", feature_value, " class: ", class_name)

                        if class_name not in feature_counts:
                            feature_counts[class_name] = {}
                            feature_probabilities[class_name] = {}

                        if feature not in feature_counts[class_name]:
                            feature_counts[class_name][feature] = {}
                            feature_probabilities[class_name][feature] = {}

                        for feature_value_1 in possible_values:
                            feature_counts[class_name][feature][feature_value_1] = 1
                            feature_probabilities[class_name][feature][feature_value_1] = 0
            print('\n')

        # print(feature_counts)

        for feature in range(num_features):
            for class_name in class_counts:
                for row in self.training_data:
                    if row[-1] == class_name:
                        feature_value = row[feature]
                        feature_counts[class_name][feature][feature_value] += 1

        # print(feature_counts)

        for class_name in feature_counts:
            for feature in range(num_features):
                q1 = feature_counts[class_name][feature].pop('q1')
                q2 = feature_counts[class_name][feature].pop('q2')
                q3 = feature_counts[class_name][feature].pop('q3')
                q4 = feature_counts[class_name][feature].pop('q4')

                # print(q1, q2, q3, q4)

                sum = q1 + q2 + q3 + q4

                feature_probabilities[class_name][feature]['q1'] = q1/sum
                feature_probabilities[class_name][feature]['q2'] = q2/sum
                feature_probabilities[class_name][feature]['q3'] = q3/sum
                feature_probabilities[class_name][feature]['q4'] = q4/sum

        print(feature_probabilities)
        self.probability_table = feature_probabilities

    def classify(self, row):
        print(row)

def main():
    soy = TenFold()

    soy.load("../data/processed_data/iris_processed.data")

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for fold in folds:
        soy_training = soy.get_training_set(fold)
        soy_testing = soy.get_test_set(fold)

        classifier = NaiveBayesClassifier(soy_training, soy_testing)
        classifier.train()

        for row in soy_testing:
            classifier.classify(row)


main()