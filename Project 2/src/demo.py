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
    classification = "Project 2\data\soybean-small.data"
    regression = "Project 2\data\\forestfires.data"

    classification_hyper = [[4]]
    regression_hyper = [[5, 0.52, 218]]


    all_predictions_edited_knn = []
    all_predictions_k_mean_knn = []
    all_predictions_knn = []

    all_actual_edited_knn = []
    all_actual_k_mean_knn = []
    all_actual_knn = []
    k = classification_hyper[0][0]

    data = Data(classification, 'class')

    # Distance example and classification of a point with k nearest neighbors
    training = data.get_training_set(1)
    test = data.get_test_set(1)

    knn = KNN(training, test)
    distance = knn.get_distance(training[0][:-1], test[0][:-1])
    print("Distance Between",training[0][:-1]," and ",test[0][:-1]," ",distance)

    input("Continue: ")
    print("k nearest neighbors")

    print("Point classification",knn.classify(training[0], k, print_distances = True))

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

    ConfusionMatrix(all_actual_edited_knn, all_predictions_edited_knn).print_confusion_matrix()
    ConfusionMatrix(all_actual_k_mean_knn, all_predictions_k_mean_knn).print_confusion_matrix()
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
    print("RBF Kernal ex: ", knn.classify(training[1],k, s))

    # TODO Demonstrate a data point being associated with a cluster while performing k-means clustering


    input("Continue: ")


    show_example = False

    for fold in folds:
        print("Fold number:", fold)

        training = data.get_training_set(fold)
        test = data.get_test_set(fold)

        # TODO Show your data being split into ten folds for one of the data sets. IDK how to do this.
        if fold == 1:
            print("Training data for fold 1:", training[0])
            print("Test data for fold 1:", test[0])
            show_example = True
        else:
            show_example = False


        # Edited KNN
        edited = EditedKNN(training, test, e, is_classification=False).edit(k, s, show_example=show_example)
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
    print("Mean Squared Error for Edited KNN:", mse_edited_knn)
    print("Mean Squared Error for K Means:", mse_k_mean_knn)
    print("Mean Squared Error for KNN:", mse_knn)

    input("Continue: ")

if __name__ == '__main__':
    main()
