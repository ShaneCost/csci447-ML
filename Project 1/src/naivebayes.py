import csv
from ten_fold import TenFold

# Used to import .csv data as a 2d array
def importcsv(file_name):
    with open(file_name, mode='r', newline='') as file:
        array = list(csv.reader(file)) 
    return array

class NaiveBayesClassifier:
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data


    # Export propability array and others
    def export_dic(self, filename ,data):
         
        # Open the file for writing
        with open(filename, 'w', newline='') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)
            
            # Write the header row
            writer.writerow(['Column', 'Class', 'Feature', 'Propability'])
            
            # Flatten and write the data
            for outer_key, categories in data.items():
                for category, questions in categories.items():
                    for question, score in questions.items():
                        writer.writerow([outer_key, category, question, score])

    class ProbabilityTable:
        def __init__(self, training_data):
            self.training_data = training_data
        
        # Creates probability table as a dictinary
        def generate_probability_table(self):
            
            # Dictinary for number of uniqe features dependong on (column) and (column by class)
            column_feature_counts = {}
            column_feature_count_by_class = {}

            # Iterate over each column index (assuming all rows have the same number of columns)
            for column_index in range(len(self.training_data[0])-1):
                feature_counts = {}
                feature_counts_by_class = {}

                # Iterate over each row
                for row in self.training_data:
                    # Get the feature for the current column
                    feature = row[column_index]
                    class_label = row[-1]

                    # Update the count for this feature in the current column
                    if feature in feature_counts:
                        feature_counts[feature] += 1
                    else:
                        feature_counts[feature] = 1

                    # Initialize dictionary for this class if not already present
                    if class_label not in feature_counts_by_class:
                        feature_counts_by_class[class_label] = {}

                    # Update the count for this feature for the current class
                    if feature in feature_counts_by_class[class_label]:
                        feature_counts_by_class[class_label][feature] += 1
                    else:
                        feature_counts_by_class[class_label][feature] = 1

                    # Add the feature counts for this column to the main dictionary
                    column_feature_counts[column_index] = feature_counts
                    column_feature_count_by_class[column_index] = feature_counts_by_class

                    # Stores list of every uniqe class
                    classes = list(set(row[-1] for row in self.training_data))
            
            # Goes throught column, class, and featue and divides each # features per class by the total # of that feature
            for column in range(0, len(column_feature_count_by_class)):
                for class_label in classes:
                    class_features = column_feature_count_by_class[column][class_label] 
                    for feature in class_features:
                        column_feature_count_by_class[column][class_label][feature] = column_feature_count_by_class[column][class_label][feature] / column_feature_counts[column][feature]


            # Column_features_count_by_class now contains the propability that each column and features belongs to that class
            return column_feature_count_by_class


def main():

    soy = TenFold()
    soy.load("Project 1\data\processed_data\soybean-small_processed.data")

    for i in range(1,11):
        soy_training = soy.get_training_set(i)
        soy_testing = soy.get_test_set(i)

        classifier = NaiveBayesClassifier(soy_training, soy_testing)

        # Instantiate ProbabilityTable with the training data
        pt = NaiveBayesClassifier.ProbabilityTable(classifier.training_data)
        data = pt.generate_probability_table()
        print(data)


if __name__ == "__main__":
    main()
