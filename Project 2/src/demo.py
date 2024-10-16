__author__ = "<Hayden Perusich>"

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

    # Windows 
    classification = "../data/soybean-small.data"
    regression = "../data/forestfires.data"

    classification_hyper = [[4]]
    regression_hyper = [[5, 0.52, 218]]

    all_predictions_edited_knn = []
    all_predictions_k_mean_knn = []
    all_predictions_knn = []

    all_actual_edited_knn = []
    all_actual_k_mean_knn = []
    all_actual_knn = []
    k = classification_hyper[0][0]

    data = Data(classification, 'class', print_folds=True)

    print("Stratified Ten Fold")
    print(data.raw_data, "\n")

    data.print_ten_folds()


    input("Continue: ")
    # Distance example and classification of a point with k nearest neighbors
    training = data.get_training_set(1)
    test = data.get_test_set(1)

    knn = KNN(training, test)
    distance = knn.get_distance(training[0][:-1], test[0][:-1])
    print("Distance Between\n",training[0][:-1],"\nand\n",test[0][:-1],"\n="
                                                                       "",distance)

    input("Continue: ")
    print("k nearest neighbors")

    print("Point classification",knn.classify(training[0], k, print_distances = True))

    input("Continue: ")

    print("K means classification")
    k_means = KMeans(training, 5, "class", print_cluster=True)

    input("Continue: ")
    for fold in folds:

        print("Fold number:", fold)
        
        training = data.get_training_set(fold)
        test = data.get_test_set(fold)

        # Edited KNN
        edited = EditedKNN(training, test).edit(k)
        edited_knn = KNN(edited.training_data, test)
        ekn_predictions = edited_knn.classify_all(k)

        all_actual_edited_knn.extend(edited_knn.get_actual_all())
        all_predictions_edited_knn.extend(ekn_predictions)

        # # K Means
        num_clusters = len(edited.training_data)
        k_means = KMeans(training, num_clusters, "class")
        training_set_for_k_means = k_means.centroid_set
        k_means_knn = KNN(training_set_for_k_means, test)
        km_predictions = k_means_knn.classify_all(k)

        all_actual_k_mean_knn.extend(k_means_knn.get_actual_all())
        all_predictions_k_mean_knn.extend(km_predictions)

         # KNN
        knn = KNN(training, test)
        knn_predictions = knn.classify_all(k)

        all_actual_knn.extend(knn.get_actual_all())
        all_predictions_knn.extend(knn_predictions)

    print("Edited KNN")
    ConfusionMatrix(all_actual_edited_knn, all_predictions_edited_knn).print_confusion_matrix()
    print("K-Means KNN")
    ConfusionMatrix(all_actual_k_mean_knn, all_predictions_k_mean_knn).print_confusion_matrix()
    print("KNN")
    ConfusionMatrix(all_actual_knn, all_predictions_knn).print_confusion_matrix()

    input("Continue: ")

    # Create data class
    all_predictions_edited_knn = []
    all_predictions_k_mean_knn = []
    all_predictions_knn = []
    
    all_actual_edited_knn = []
    all_actual_k_mean_knn = []
    all_actual_knn = [] 
    k = regression_hyper[0][0]
    s = regression_hyper[0][1]
    e = regression_hyper[0][2]
    
    # RBF Kernal exmaple and prediciton of a point for regression
    data = Data(regression, "regress")
    
    training = data.get_training_set(1)
    test = data.get_test_set(1)

    knn = KNN(training, test, is_classification=False)
   

    print("k nearest neighbors")
    knn_predictions = knn.classify(training[0],k, s, print_distances = True)
    print("Predicaiton: ",knn_predictions)

    input("Continue: ")
    print("RBF Kernal ex: ", knn.classify(training[1],k, s, print_rbf=True
                                          ))
    input("Continue: ")


    show_example = False

    for fold in folds:
        print("Fold number:", fold)

        training = data.get_training_set(fold)
        test = data.get_test_set(fold)

        # Edited KNN
        edited = EditedKNN(training, test, e, is_classification=False).edit(k, s, show_example=True)
        edited_knn = KNN(edited.training_data, test)
        ekn_predictions = edited_knn.classify_all(k, s)

        all_actual_edited_knn.extend(edited_knn.get_actual_all())
        all_predictions_edited_knn.extend(ekn_predictions)

        # K Means
        num_clusters = len(edited.training_data)
        k_means = KMeans(training, num_clusters, "regress")
        training_set_for_k_means = k_means.centroid_set
        k_means_knn = KNN(training_set_for_k_means, test, is_classification=False)
        km_predictions = k_means_knn.classify_all(k, s)

        all_actual_k_mean_knn.extend(k_means_knn.get_actual_all())
        all_predictions_k_mean_knn.extend(km_predictions)

        # KNN
        knn = KNN(training, test, is_classification=False)
        knn_predictions = knn.classify_all(k, s)

        all_actual_knn.extend(knn.get_actual_all())
        all_predictions_knn.extend(knn_predictions)

    # Calculate Mean Squared Error for each model
    mse_edited_knn = Loss(all_predictions_edited_knn, all_actual_edited_knn, "regress", e).mean_squared_error()
    mse_k_mean_knn = Loss(all_predictions_k_mean_knn, all_actual_k_mean_knn, "regress", e).mean_squared_error()
    mse_knn = Loss(all_predictions_knn, all_actual_knn, "regress", e).mean_squared_error()

    # Output the results
    input("Continue: ")
    print("Mean Squared Error for Edited KNN:", mse_edited_knn)
    print("Mean Squared Error for K Means:", mse_k_mean_knn)
    print("Mean Squared Error for KNN:", mse_knn)

    input("Continue: ")

if __name__ == '__main__':
    main()
