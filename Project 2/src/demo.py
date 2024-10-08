from data import *
from knn import *
from edited_knn import *
from k_means import *
from confusion_matrix import *
from loss import *

# CONSTANTS
BREASTCANCER_K = 3

GLASS_K = 2

SOYBEAN_K = 4

FORESTFIRES_K = 5
FORESTFIRES_S = 0.52
FORESTFIRES_E  = 218.168

MACHINE_K = 1
MACHINE_S = 0.86
MACHINE_E = 244.6

ABALONE_K = 10
ABALONE_S = 0.72
ABALONE_E = 0.28

def main():
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    classification = ["Project 2\data\\breast-cancer-wisconsin.data","Project 2\data\glass.data","Project 2\data\soybean-small.data"]
    regression = ["Project 2\data\\forestfires.data","Project 2\data\machine.data","Project 2\data\\abalone.data"]

    classification_hyper = [[3], [2], [4]]
    regression_hyper = [[5, 0.52, 218.168], [1, 0.52, 218.168], [10, 0, 0]]

    # classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]
    # regression = ["../data/forestfires.data", "../data/machine.data", "../data/abalone.data",]

    # DEMONSTRATION OF USING k_means TO GET REDUCED TRAINING SET
    for index, file in enumerate(regression):
        # Create data class
        data = Data(file, "regress")    

        all_predictions = []
        all_actual = []

        # Iterate over the folds
        for fold in folds:
            # Get training and test set
            training = data.get_training_set(fold)
            test = data.get_test_set(fold)

            # Use editedKNN to derive the number of clusters
            k = regression_hyper[index][0]
            s = regression_hyper[index][1]
            e = regression_hyper[index][2]

            # edited = EditedKNN(training, test, e, is_classification=False).edit(k, s) 
            # print("Done getting data")

            # num_clusters = len(edited.training_data)
            # k_means = KMeans(training, num_clusters, "regress")
            # training_set_for_k_means = k_means.centroid_set
 
    #         # Instantiate knn with reduced training set and call classify_all()
            knn = KNN(training, test)

            print(knn.get_actual_all())
            knn_c = knn.classify_all(k, s)
            print(knn_c)

            all_actual.extend(knn.get_actual_all())
            all_predictions.extend(knn_c)
            print(fold)
            print(Loss(all_predictions, all_actual, "regress", e).mean_squared_error())


        print(Loss(all_predictions, all_actual, "regress", e).mean_squared_error())

    


    # for file in regression:
    #     data = Data(file, "regress")

if __name__ == '__main__':
    main()

