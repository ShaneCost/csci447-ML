from data import *
from knn import *
from edited_knn import *
from k_means import *

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
ABALONE_S = 0
ABALONE_E = 0

def main():
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    classification = ["../data/breast-cancer-wisconsin.data", "../data/glass.data", "../data/soybean-small.data"]
    regression = ["../data/forestfires.data", "../data/machine.data", "../data/abalone.data",]

    # DEMONSTRATION OF USING k_means TO GET REDUCED TRAINING SET
    for file in classification:
        # Create data class
        data = Data(file, "class")

        # Iterate over the folds
        for fold in folds:
            # Get training and test set
            training = data.get_training_set(fold)
            test = data.get_test_set(fold)

            # Use editedKNN to derive the number of clusters
            # edited = EditedKNN(training, test, 0, True).edit(1) # TODO: UPDATE edit() TO TAKE IN CORRECT k VALUE BY FILE
            # num_clusters = len(edited.training_data)
            num_clusters = 2
            # Instantiate k_means # TODO: I was getting an error when using the edited_knn() class (hence it being commented out above) because of a missing parameter value in one of the functions. I didn't want to change the file because I thought you might have already fixed the error but not pushed the changes. Once the error is fixed, uncomment the 2 lines above to set num_cluster = size of the edited data set instead of the hard coded '2'
            k_means = KMeans(training, num_clusters, "class")

            # Use the centroid set as the new training set
            training_set_for_k_means = k_means.centroid_set

            # Instantiate knn with reduced training set and call classify_all()
            knn = KNN(training_set_for_k_means, test, True)
            predicted = knn.classify_all(1, 1) # TODO: UPDATE classify_all() TO TAKE IN CORRECT k AND sigma VALUES BASED ON FILE

    for file in regression:
        data = Data(file, "regress")

if __name__ == '__main__':
    main()

