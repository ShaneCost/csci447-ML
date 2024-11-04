__author__ = "<Hayden Perusich>"

from feedforward_network import *
from loss import *
from root_data import *
from meta_data import *
import os
## HYPERPARAMETERS
# soybean: HIDDEN_LAYERS 0-2, NODES _, LEARNING_RATE_, BATCH_SIZE_,   

tuned_hyperparameters_classification = {
    'breast-cancer-wisconsin' : {
        0: {
            'num_nodes': 68,
            'learning_rate': 0.0183681632653
        },
        1: {
            'num_nodes': 43,
            'learning_rate': 0.01223143323
        },
        2: {
            'num_nodes': 34,
            'learning_rate': 0.01967512316
        },
    },
    'glass': {
        0:{
            'num_nodes': 82,
            'learning_rate': 0.006123387755
        },
        1: {
            'num_nodes': 61,
            'learning_rate': 0.002231523367
        },
        2: {
            'num_nodes': 47,
            'learning_rate': 0.00967512316
        },
    },
    'soybean-small': {
        0: {
            'num_nodes': 31,
            'learning_rate': 0.0244905510204
        },
        1: {
            'num_nodes': 25,
            'learning_rate': 0.01976302731
        },
        2: {
            'num_nodes': 23,
            'learning_rate': 0.0364031483
        }
    }
}


tuned_hyperparameters_regression= {
    'forestfires' : {
        0: {
            'num_nodes': 26,
            'learning_rate': 0.065306469
        },
        1: {
            'num_nodes': 11,
            'learning_rate': 0.041283924
        },
        2: {
            'num_nodes': 10,
            'learning_rate': 0.0125829273
        },
    },
    'machine': {
        0:{
            'num_nodes': 34,
            'learning_rate': 0.00273920
        },
        1: {
            'num_nodes': 9,
            'learning_rate': 0.04940382
        },
        2: {
            'num_nodes': 7,
            'learning_rate': 0.01382018
        },
    },
    'abalone': {
        0: {
            'num_nodes': 11,
            'learning_rate': 0.0244905
        },
        1: {
            'num_nodes': 5,
            'learning_rate': 0.02940204
        },
        2: {
            'num_nodes': 4,
            'learning_rate': 0.014201739
        }
    }
}


def main():
    #"Project 3\data\soybean-small.data"
    # classification = [ "Project 3\data\glass.data","Project 3\data\\breast-cancer-wisconsin.data"]
    regression = ["Project 3\data\\forestfires.data", "Project 3\data\machine.data", "Project 3\data\\abalone.data", ]
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for file in regression: 
        data = RootData(file, False)
        filename = os.path.splitext(os.path.basename(file))[0]
        for num_hiddent_layers in range(3):
            all_predictions = []
            all_actual = []

            for fold in folds:
                print("Started with fold:",fold, " hidden layers:",num_hiddent_layers, filename)

                training = MetaData(data.get_training_set(fold))
                test = MetaData(data.get_test_set(fold))

                hidden_size = tuned_hyperparameters_regression[filename][num_hiddent_layers]["num_nodes"]
                learning_rate = tuned_hyperparameters_regression[filename][num_hiddent_layers]["learning_rate"]
                
                ffn = FeedForwardNetwork(training, test, num_hiddent_layers, hidden_size=hidden_size,
                                 input_size=data.num_features, output_size=1,
                                 learning_rate=learning_rate, is_class=False)

                ffn.train()
                prediction, actual  = ffn.test()
                all_predictions.extend(prediction)
                all_actual.extend(actual)
                print("Done with fold", fold, " : ", filename)

            print("hidden layers: ",num_hiddent_layers, filename)
            loss = Loss(prediction, actual, is_class=False).mean_squared_error()
            print(loss)

# main()



    





