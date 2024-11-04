__author__ = "<Shane Costello>"

import random
from root_data import *
from meta_data import *
# from feedforward_shane import *
from feedforward_network import *
from loss import *

def tune(data, num_hidden_layers):
    """
    Method used to tune a set of hyperparameters

    :param data: Instance of the RootData class
    :param num_hidden_layers: Integer representing the number of hidden layers in our network
    """
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Extract the hyperparameters from the RootData class
    num_nodes = data.hyperparameters['num_nodes']
    learning_rate = data.hyperparameters['learning_rate']

    # Begin tuning the number of nodes per hidden layer
    while not num_nodes.is_tuned:
        score = 0
        # Iterate over the set of folds
        for fold in folds:
            # Generate a training set with a holdout fold
            training = MetaData(data.get_training_set(fold))

            # Extract the tuning (testing) set
            testing = MetaData(data.tuning)

            # Initialize an instance of the FeedForwardNetwork
            ffn = FeedForwardNetwork(training, testing, num_hidden_layers, hidden_size=int(num_nodes.value),
                                 input_size=data.num_features, output_size=1,
                                 learning_rate=0.01, is_class=False)
            # Train the model
            ffn.train()

            # Test against the model
            predicted, actual = ffn.test()

            # Generate a loss function value
            loss = Loss(predicted, actual, is_class=False)

            # Aggregate loss function results
            score += loss.results

        # Average loss function results across the ten folds
        score /= 10

        # Update the hyperparameter value, and continue tuning
        num_nodes.update(score)

    # Begin tuning the learning rate value
    while not learning_rate.is_tuned:
        score = 0
        # Iterate over the set of folds
        for fold in folds:
            # Generate a training set with a holdout fold
            training = MetaData(data.get_training_set(fold))

            # Extract the tuning (testing) set
            testing = MetaData(data.tuning)

            # Initialize an instance of the FeedForwardNetwork
            ffn = FeedForwardNetwork(training, testing, num_hidden_layers, hidden_size=int(num_nodes.value),
                                 input_size=data.num_features, output_size=1,
                                 learning_rate=learning_rate.value, is_class=False)
            # Train the model
            ffn.train()

            # Test against the model
            predicted, actual = ffn.test()

            # Generate a loss function value
            loss = Loss(predicted, actual, is_class=False)

            # Aggregate loss function results
            score += loss.results

        # Average loss function results across the ten folds
        score /= 10

        # Update the hyperparameter value, and continue tuning
        learning_rate.update(score)

def main():
    num_hidden_layers = [0, 1, 2]
    regression = ["Project 3/data/forestfires.data", "Project 3/data/machine.data", "Project 3/data/abalone.data"]

    for file in regression:
        data = RootData(file, False)
        for layer_size in num_hidden_layers:
            tune(data, layer_size)
            print("name: ", data.name)
            print("num hidden layers: ", layer_size)
            print("num nodes: ", data.hyperparameters['num_nodes'].value)
            print("learning rate: ", data.hyperparameters['learning_rate'].value)
            print('\n')

# main()