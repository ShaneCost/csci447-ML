__author__ = "<Shane Costello>"

from ten_fold import TenFold
from naive_bayes import NaiveBayesClassifier
from confusion_matrix import ConfusionMatrix

def run(file_path):
    # Creat a TenFold object to store our data
    data = TenFold()
    data.load(file_path)

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Arrays to store actual + predicted class values
    actual_values = []
    predicted_values = []

    # Iterate over each fold where the current fold = test set
    for fold in folds:
        training = data.get_training_set(fold) # Get the training set
        testing = data.get_test_set(fold) # Get test set

        # Creat instance of NaiveBayesClassifier
        classifier = NaiveBayesClassifier(training, testing)
        classifier.train() # train model on our training data

        # Iterate through each test value in testing set
        for row in testing:
            prediction = classifier.classify(row)

            predicted_values.append(classifier.arg_max(prediction))
            actual_values.append(row[-1])

    # Create confusion matrix based on predicted vs. actual values
    confusion_matrix = ConfusionMatrix(actual_values, predicted_values)
    confusion_matrix.print_confusion_matrix()
    return confusion_matrix

def main():
    clean_files = ["../data/processed_data/iris_processed.data",
             "../data/processed_data/breast-cancer-wisconsin_processed.data",
             "../data/processed_data/glass_processed.data",
             "../data/processed_data/soybean-small_processed.data",
             "../data/processed_data/house-votes-84_processed.data"]

    noisy_files = ["../data/noisy_data/iris_noisy.data",
             "../data/noisy_data/breast-cancer-wisconsin_noisy.data",
             "../data/noisy_data/glass_noisy.data",
             "../data/noisy_data/soybean-small_noisy.data",
             "../data/noisy_data/house-votes-84_noisy.data"]

    clean_scores = []
    for file in clean_files:
        clean_scores.append(run(file).scores)

    noisy_scores = []
    for file in noisy_files:
        noisy_scores.append(run(file).scores)

    print(clean_scores)
    print(noisy_scores)

main()