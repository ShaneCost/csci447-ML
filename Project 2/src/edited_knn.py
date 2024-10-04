from knn import KNN

class EditedKNN(KNN):

    def __init__(self, data, fold, is_classification=True):
        
        super().__init__(data, fold, is_classification)
    
    def edit(self, k, p):

        new_training_data = []

        for point in self.training_data:
        
            if self.is_classification:
                correct_classification = (self.classify(point, k, p) == self.get_actual(point))
            if not self.is_classification:
                correct_classification = () 
            
            
            if correct_classification:
                new_training_data.append(point)
 
            if len(new_training_data) >= 0.2 * len(self.training_data):
                break

        self.training_data = new_training_data



from data import Data
def main():

    path = "Project 2\data\machine.data"
    data = Data(path, "regress")

    print(data.hyperparameters['epsilon'].value)


    edited_knn = EditedKNN(data, 1)
    
    # predications1 = edited_knn.classify_all(2,2)
    # print(len(edited_knn.training_data))
    
    # edited_knn.edit(2, 2)

    # print(len(edited_knn.training_data))

    # predications2 = edited_knn.classify_all(2,2)
    # print(len(predications2))


    # difference = [item for item in predications1 if item not in predications2]
    # print(difference)
main()