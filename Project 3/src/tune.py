import random
from root_data import *
from meta_data import *
from feedforward_shane import *
from loss import *

def tune(data, num_hidden_layers):
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    num_nodes = data.hyperparameters['num_nodes']
    learning_rate = data.hyperparameters['learning_rate']

    while not num_nodes.is_tuned:
        score = 0
        print('\t\tcurrent node value = ', int(num_nodes.value))
        for fold in folds:
            training = MetaData(data.get_training_set(fold))
            testing = MetaData(data.tuning)
            ffn = FeedForwardNetwork(training, testing, num_hidden_layers, int(num_nodes.value), data.num_features,
                                     data.num_classes, data.classes, 0.01, data.is_class)
            # print('\t\t\ttraining')
            ffn.train()
            # print('\t\t\ttesting on fold ', fold)
            predicted, actual = ffn.test()
            loss = Loss(predicted, actual)
            score += loss.results
        score /= 10
        num_nodes.update(score)

    while not learning_rate.is_tuned:
        score = 0
        print('\t\tcurrent learning rate value = ', int(learning_rate.value))
        for fold in folds:
            training = MetaData(data.get_training_set(fold))
            testing = MetaData(data.tuning)
            ffn = FeedForwardNetwork(training, testing, num_hidden_layers, data.num_features, data.num_features,
                                            data.num_classes, data.classes, learning_rate.value, data.is_class)
            # print('\t\t\ttraining')
            ffn.train()
            # print('\t\t\ttesting on fold ', fold)
            predicted, actual = ffn.test()
            loss = Loss(predicted, actual)
            score += loss.results
        score /= 10
        learning_rate.update(score)

def main():
    num_hidden_layers = [0, 1, 2]
    # All data set used for classification problems
    classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]
    # All data set used for regression problems
    regression = ["../data/forestfires.data", "../data/machine.data", "../data/abalone.data", ]

    for file in classification:
        print("tuning classification")
        data = RootData(file)
        for layer_size in num_hidden_layers:
            tune(data, layer_size)
            print('\nRESULTS')
            print("name: ", data.name)
            print("num hidden layers: ", layer_size)
            print("num nodes: ", data.hyperparameters['num_nodes'].value)
            print("learning rate: ", data.hyperparameters['learning_rate'].value)
            print('\n')

    # for file in regression:
    #     data = RootData(file, False)
    #     for layer_size in num_hidden_layers:
    #         tune(data, layer_size)
    #         print("name: ", data.name)
    #         print("num hidden layers: ", layer_size)
    #         print("num nodes: ", data.hyperparameters['num_nodes'].value)
    #         print("learning rate: ", data.hyperparameters['learning_rate'].value)
    #         print('\n')


main()