from knn import KNN

class EditedKNN(KNN):

    def __init__(self, training_set, test_set, epsilon, is_classification=True):
        
        super().__init__(training_set, test_set, is_classification)
        self.epsilon = epsilon
    
    def edit(self, k):

        new_training_data = []

        for point in self.training_data:
            
            # determining correct_classification 
            if self.is_classification:
                correct_classification = (self.classify(point, k) == self.get_actual(point))

            if not self.is_classification:
                largest_value = self.classify(point, k) + self.epsilon
                smallest_value = self.classify(point, k) - self.epsilon
                correct_classification = largest_value >=  self.get_actual(point) <= smallest_value 

            if correct_classification:
                new_training_data.append(point)
 
            if len(new_training_data) >= 0.2 * len(self.training_data):
                break

        self.training_data = new_training_data



from data import Data
def main():

    path = "Project 2\data\machine.data"
    data = Data(path, "regress")
    
    training_set = data.get_training_set(1)
    test_set = data.get_test_set(1)


    epsilon =  data.hyperparameters['epsilon'].value

    edited_knn = EditedKNN(training_set, test_set, epsilon)
    
    predications1 = edited_knn.classify_all(2)
    print(predications1)
    print(len(edited_knn.training_data))

    edited_knn.edit(2)

    print(len(edited_knn.training_data))

    predications2 = edited_knn.classify_all(2)
    print(predications2)

main()