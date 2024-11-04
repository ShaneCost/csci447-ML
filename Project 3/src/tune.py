import random
from root_data import *
from meta_data import *
# from feedforward_shane import *
from feedforward_network import *
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
            
            ffn = FeedForwardNetwork(training, testing, num_hidden_layers, hidden_size=int(num_nodes.value),
                                 input_size=data.num_features, output_size=1,
                                 learning_rate=0.01, is_class=False)
                                     
            print('\t\t\ttraining')
            ffn.train()
            print('\t\t\ttesting on fold ', fold)
            predicted, actual = ffn.test()
          
            loss = Loss(predicted, actual, is_class=False)
            score += loss.results
        score /= 10
        num_nodes.update(score)

    while not learning_rate.is_tuned:
        score = 0
        print('\t\tcurrent learning rate value = ', int(learning_rate.value))
        for fold in folds:
            training = MetaData(data.get_training_set(fold))
            testing = MetaData(data.tuning)
            ffn = FeedForwardNetwork(training, testing, num_hidden_layers, hidden_size=int(num_nodes.value),
                                 input_size=data.num_features, output_size=1,
                                 learning_rate=learning_rate.value, is_class=False)
            print('\t\t\ttraining')
            ffn.train()
            print('\t\t\ttesting on fold ', fold)
            predicted, actual = ffn.test()
            loss = Loss(predicted, actual, is_class=False)
            score += loss.results
        score /= 10
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

            # You may want to test the performance metrics here instead of confusion matrix
            # predicted, actual = ffn.test()
            # Calculate performance metrics like MSE or RMSE instead
            # loss = Loss(predicted, actual)  # Adjust as per your loss handling

main()