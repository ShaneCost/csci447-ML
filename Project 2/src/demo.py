from data import *
from k_means import *

def main():

    classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]
    regression = ["../data/abalone.data", "../data/forestfires.data", "../data/machine.data"]

    for file in classification:
        data = Data(file, "class")

        training_set = data.get_training_set(10)
        test_set = data.get_test_set(10)

        k_means = KMeans(training_set, 5)

    for file in regression:
        data = Data(file, "regress")
        print(data.name)
        for hyperparameter in data.hyperparameters:
            print(hyperparameter, data.hyperparameters[hyperparameter].value)
        print('\n')

main()
