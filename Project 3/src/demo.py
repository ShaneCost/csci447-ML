__author__ = "<Hayden Perusich>"

from feedforward_network import *
from loss import *
from confusion_matrix import *
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
            'num_nodes': 20,#43,
            'learning_rate': 0.01223143323
        },
        2: {
            'num_nodes': 10,#34,
            'learning_rate': 0.01967512316
        },
    },
    'glass': {
        0:{
            'num_nodes': 80,
            'learning_rate': 0.006123387755
        },
        1: {
            'num_nodes': 20,
            'learning_rate': 0.002231523367
        },
        2: {
            'num_nodes': 10,
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


import os
from feedforward_network import *
from meta_data import *
from root_data import *
from confusion_matrix import *

def main():
    # List of classification datasets
    classification = [
        "Project 3/data/glass.data",
        "Project 3/data/glass.data",
        "Project 3/data/soybean-small.data"
    ]
    
    # Folds for cross-validation
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Loop through each classification dataset
    for file in classification:
        data = RootData(file, True)
        filename = os.path.splitext(os.path.basename(file))[0]  # Extract dataset name without extension

        # Loop through different numbers of hidden layers (0, 1, 2 hidden layers)
        for num_hidden_layers in range(3):
            all_predictions = []
            all_actual = []

            # Perform cross-validation for each fold
            for fold in folds:
                print(f"Started with fold: {fold}, hidden layers: {num_hidden_layers}, dataset: {filename}")

                # Prepare training and test data for the current fold
                training = MetaData(data.get_training_set(fold))
                test = MetaData(data.get_test_set(fold))

                # Get the hyperparameters for the current dataset and hidden layers configuration
                hidden_size = tuned_hyperparameters_classification[filename][num_hidden_layers]["num_nodes"]
                learning_rate = tuned_hyperparameters_classification[filename][num_hidden_layers]["learning_rate"]
                
                # Create and train the FeedForwardNetwork
                ffn = FeedForwardNetwork(
                    training, test, num_hidden_layers, hidden_size,
                    data.num_features, data.num_classes, data.classes, learning_rate
                )
                ffn.train()

                # Get predictions and actual values from the test set
                prediction, actual = ffn.test()

                # Collect predictions and actuals for the confusion matrix
                all_predictions.extend(prediction)
                all_actual.extend(actual)

                print(f"Done with fold {fold} : {filename}")

            # After completing all folds, print the confusion matrix
            ConfusionMatrix(all_actual, all_predictions).print_confusion_matrix()

# Entry point for running the script
if __name__ == "__main__":
    main()





