__author__ = "<Shane Costello>"

from data import *
from knn import *
from loss import *

def tune(data, hyperparameters, class_or_regress):
    """
    Function used to tune the set of hyperparameters

    :param data: full data set
    :param hyperparameters: a set of instances of hyperparameter class
    :param class_or_regress: string denoting whether it is a classification or regression problem
    :return: Tuned hyperparameters
    """
    k = hyperparameters['k']
    epsilon = hyperparameters['epsilon']
    sigma = hyperparameters['sigma']

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tuning = data.tuning

    if class_or_regress == "class":
        is_class = True
    else:
        is_class = False

    # Tune K
    while not k.is_tuned:
        score = 0
        for fold in folds: # Iterate over the set of folds
            training_set = data.get_training_set(fold) # Get a training set
            knn = KNN(training_set, tuning, is_class) # Use the tuning set as the test set
            predictions = knn.classify_all(k.value, sigma.value) # Get predicted values
            actual = knn.get_actual_all() # Get actual values
            loss = Loss(actual, predictions, class_or_regress, epsilon.value) # Use loss function to derive performance metric
            score += loss.results
        score /= 10 # Average scores across holdout folds
        k.step(score) # Step up K value and check if it is properly tuned

    # If it is a regression problem
    if class_or_regress == "regress":
        # Tune Sigma
        while not sigma.is_tuned:
            score = 0
            print("\tcurr sigma: ", sigma.value)
            for fold in folds:
                training_set = data.get_training_set(fold)
                knn = KNN(training_set, tuning, is_class)
                predictions = knn.classify_all(k.value, sigma.value)
                actual = knn.get_actual_all()
                loss = Loss(actual, predictions, class_or_regress, epsilon.value)
                score += loss.results
            score /= 10
            sigma.step(score)

        # Tune epsilon
        while not epsilon.is_tuned:
            score = 0
            print("\tcurr epsilon: ", epsilon.value)
            for fold in folds:
                training_set = data.get_training_set(fold)
                knn = KNN(training_set, tuning, is_class)
                predictions = knn.classify_all(k.value, sigma.value)
                actual = knn.get_actual_all()
                loss = Loss(actual, predictions, class_or_regress, epsilon.value)
                score += loss.results
            score /= 10
            epsilon.step(score)


def main():

    # All data set used for classification problems
    classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]
    # All data set used for regression problems
    regression = ["../data/forestfires.data", "../data/machine.data", "../data/abalone.data",]

    for file in classification:
        data = Data(file, "class")
        print("\n", data.name)
        tune(data, data.hyperparameters, "class")

    for file in regression:
        data = Data(file, "regress")
        print("\n", data.name)
        tune(data, data.hyperparameters, "regress")

    data = Data("../data/abalone.data", "regress")
    tune(data, data.hyperparameters, "regress")

main()
