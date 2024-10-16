__author__ = "<Hayden Perusich>"

from knn import KNN

class EditedKNN(KNN):
    def __init__(self, training_set, test_set, epsilon=0, is_classification=True):
        # Initialize the base KNN class with training and test sets
        super().__init__(training_set, test_set, is_classification)
        # Set epsilon for regression tasks to determine acceptable prediction range
        self.epsilon = epsilon
    
    def edit(self, k, s=0, show_example=False):
        # Create a new list to hold edited training data
        new_training_data = []

        removed_count = 0

        for point in self.training_data:
            # Determine if the current point is correctly classified
            if self.is_classification:
                correct_classification = (self.classify(point, k) == self.get_actual(point))
            else:
                # For regression, check if the actual value is within the epsilon range of the predicted value
                predicted_value = float(self.classify(point, k, s))
                correct_classification = (self.get_actual(point) >= (predicted_value - self.epsilon) and
                                          self.get_actual(point) <= (predicted_value + self.epsilon))

            # If the classification is correct, keep the point in the new training data
            if correct_classification:
                new_training_data.append(point)
            
            # If the classification is incorrect and showing examples is enabled, print the point
            if not correct_classification and show_example:
                print("The data point ", point, "is being edited out of the dataset because of incorrect classification")
                show_example = False  # Reset show_example to avoid repeated printing

            # Stop editing if we've kept at least 20% of the original training data
            if len(new_training_data) >= 0.2 * len(self.training_data):
                break

        # Update the training data to the new edited set
        self.training_data = new_training_data
        return self

# Sample usage commented out
# from data import Data
# def main():
#     path = "Project 2\data\machine.data"
#     data = Data(path, "regress")
#
#     training_set = data.get_training_set(1)
#     test_set = data.get_test_set(1)
#
#     epsilon = data.hyperparameters['epsilon'].value
#
#     edited_knn = EditedKNN(training_set, test_set, epsilon)
#
#     # Classify using the original training set
#     predictions1 = edited_knn.classify_all(2)
#     print(predictions1)
#     print(len(edited_knn.training_data))
#
#     # Edit the training set
#     edited_knn.edit(2)
#     print(len(edited_knn.training_data))
#
#     # Classify again using the edited training set
#     predictions2 = edited_knn.class
