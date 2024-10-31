from feedforward_shane import *
## HYPERPARAMETERS
# soybean: HIDDEN_LAYERS 0-2, NODES _, LEARNING_RATE_, BATCH_SIZE_,   

tuned_hyperparameters = {
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
            'learning_rate': .01976302731
        },
        2: {
            'num_nodes': 23,
            'learning_rate': .0364031483
        }
    }
}

def main():
    classification = ["Project 3\data\soybean-small.data","Project 3\data\\breast-cancer-wisconsin.data", "Project 3\data\glass.data"]
    regression = ["Project 3\data\\forestfires.data", "Project 3\data\machine.data", "Project 3\data\\abalone.data", ]
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    
    for file in classification:
        data = RootData(file)

        for fold in folds:

            training = MetaData(data.get_training_set(fold))
            test = MetaData(data.get_test_set(fold))

            ffn = FeedForwardNetwork(training, test, 1, 5, data.num_features, data.num_classes, data.classes, 0.01)

            ffn.train()
            ffn.test()

        ## print CM for each data_set
    
    for file in regression:
        data = RootData(file, False)

        for fold in folds:

            training = MetaData(data.get_training_set(fold))
            test = MetaData(data.get_test_set(fold))

            ffn = FeedForwardNetwork(training, test, 1, 5, data.num_features, 1, data.classes, 0.01)

            
            ffn.train()
            ffn.test()

    # print mean squared error for each dataset
    




    





