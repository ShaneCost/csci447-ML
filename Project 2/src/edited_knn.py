from knn import KNN


class EditedKNN(KNN):

    def __init__(self, edited_folds, tuning_set):
        self.edited_folds = edited_folds
        self.tuning_set = tuning_set
    
    def edit(self):
        k = 1
        p = 2  
        for point in self.training_data:
            correct_classification = (self.classify(point, self.training_data, k, p) == self.get_actual(point))
            if (correct_classification):
                self.training_data = self.training_data.remove(point)


def main():

    print("hi")

main()