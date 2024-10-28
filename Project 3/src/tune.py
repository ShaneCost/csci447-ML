import random
from root_data import *
from meta_data import *
from feedforward_shane import *

def tune(data):
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    num_nodes = data.hyperparameters['num_nodes']
    learning_rate = data.hyperparameters['learning_rate']
    batch_size = data.hyperparameters['batch_size']

    while not num_nodes.is_tuned:
        score = 0
        for fold in folds:
            training = MetaData(data.get_training_set(fold))
            testing = MetaData(data.tuning)
            ffn = FeedForwardNetwork(training, testing, 1, int(num_nodes.value), data.num_features, data.num_classes, data.classes, 0.01, data.is_class)
            ffn.train()
            ffn.test()
            # score += random.randint(-5, 5)
        score /= 10
        num_nodes.update(score)
        print(score)

    print(num_nodes.value)

    while not learning_rate.is_tuned:
        score = 0
        for fold in folds:
            training = MetaData(data.get_training_set(fold))
            testing = MetaData(data.tuning)
            score += random.randint(-5, 5)
        score /= 10
        learning_rate.update(score)

    while not batch_size.is_tuned:
        score = 0
        for fold in folds:
            training = MetaData(data.get_training_set(fold))
            testing = MetaData(data.get_test_set(fold))
            score += random.randint(-5, 5)
        score /= 10
        batch_size.update(score)

def main():
    # All data set used for classification problems
    classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]
    # All data set used for regression problems
    regression = ["../data/forestfires.data", "../data/machine.data", "../data/abalone.data", ]

    for file in classification:
        data = RootData(file)
        tune(data)

    for file in regression:
        data = RootData(file, False)
        tune(data)

main()