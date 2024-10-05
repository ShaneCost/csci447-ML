__author__ = "<Shane Costello>"

from data import *
from knn import *
from loss import *

def tune(data, hyperparameters, class_or_regress):
    k = hyperparameters['k']
    epsilon = hyperparameters['epsilon']
    sigma = hyperparameters['sigma']

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tuning = data.tuning

    if class_or_regress == "class":
        is_class = True
    else:
        is_class = False

    print("max k: ", k.max_value)
    while not k.is_tuned:
        score = 0
        print("\tcurr k: ", k.value)
        for fold in folds:
            training_set = data.get_training_set(fold)
            knn = KNN(training_set, tuning, is_class)
            predictions = knn.classify_all(k.value, sigma.value)
            actual = knn.get_actual_all()
            loss = Loss(actual, predictions, class_or_regress, epsilon.value)
            score += loss.results
        score /= 10
        k.step(score)
        print("\t\t score: ", score)

    print("K tuned:", k.value)

    if class_or_regress == "regress":
        print("max sigma: ", sigma.max_value)
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
            print("\t\t score: ", score)

        print("sigma tuned:", sigma.value)

        print("max epsilon: ", epsilon.max_value)
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
            print("\t\t score: ", score)

        print("epsilon tuned:", epsilon.value)

def main():

    classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]
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
