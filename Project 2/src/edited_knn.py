from knn import KNN

class EditedKNN(KNN):

    def __init__(self, training_data, testing_data, is_classification=True):
        super().__init__(training_data, testing_data, is_classification)
    
    def edit(self, k, p):

        new_training_data = []

        for point in self.training_data:
            correct_classification = (self.classify(point, k, p) == self.get_actual(point))
            if correct_classification:
                new_training_data.append(point)
 
            if len(new_training_data) >= 0.2 * len(self.training_data):
                break

        self.training_data = new_training_data



from data import Data
def main():

    path = "Project 2\data\\breast-cancer-wisconsin.data"
    data = Data(path, "class")

    training_set = data.get_training_set(1)
    test_set = data.get_test_set(1)

    edited_knn = EditedKNN(training_set, test_set)
    
    predications1 = edited_knn.classify_all(2,2)
    
    edited_knn.edit(2, 2)

    predications2 = edited_knn.classify_all(2,2)

    difference = [item for item in predications1 if item not in predications2]
    print(difference)
main()