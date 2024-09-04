import csv

# Used to import .csv data as a 2d array
def importcsv(file_name):
    with open(file_name, mode='r', newline='') as file:
        array = list(csv.reader(file)) 
    return array

class NaiveBayesClassifier:
    def __init__(self, training_file, testing_file):
        self.training_file = training_file
        self.testing_file = testing_file
        self.training_data = None
        self.testing_data = None

    # Import training and testing data as seperate datasets
    def import_training(self):
        self.training_data = importcsv(self.training_file)

    def import_testing(self):
        self.testing_data = importcsv(self.testing_file)

    # Export propability array and others
    def export_array(self):
        pass

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

                    classes = list(set(row[-1] for row in self.training_data))


            print(classes)
                


            # print(column_feature_count_by_class)
            # print(column_feature_counts)

            


def main():
    classifier = NaiveBayesClassifier('./preprocessed_data/soybean-small_processed.data', './data/soybean-small.data')
    classifier.import_training()
    classifier.import_testing()
    
    # Instantiate ProbabilityTable with the training data
    pt = NaiveBayesClassifier.ProbabilityTable(classifier.training_data)
    pt.generate_probability_table()

if __name__ == "__main__":
    main()
