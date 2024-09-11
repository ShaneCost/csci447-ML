__author__ = "<Shane Costello>"

class ConfusionMatrix:

    def __init__(self, actual, predicted):
        self.actual_values = actual
        self.predicted_values = predicted

        # Create a mapping from string labels to numeric indices
        self.classes = sorted(list(set(actual + predicted)))  # Unique sorted class labels
        self.size = len(self.classes)
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}

        # Initialize confusion matrix with zeros
        self.confusion_matrix = [[0 for _ in range(self.size)] for _ in range(self.size)]

        # Generate confusion matrix
        self.generate_confusion_matrix()

        # Generate TP, TN, FP, and FN values
        self.truth_values = self.query_matrix()

        # Generate accuracy, precision, and recall scores
        self.scores = self.generate_scores()

    def generate_confusion_matrix(self):
        """
        Populate the confusion matrix by comparing actual and predicted values.
        Each row corresponds to the actual class, and each column corresponds to the predicted class.
        """
        for actual, predicted in zip(self.actual_values, self.predicted_values):
            actual_idx = self.label_to_index[actual]  # Convert actual label to index
            predicted_idx = self.label_to_index[predicted]  # Convert predicted label to index
            self.confusion_matrix[actual_idx][predicted_idx] += 1

    def print_confusion_matrix(self):
        """
        Print the confusion matrix with labels for better understanding.
        """
        # Print the header with class labels
        print("Confusion Matrix:")
        print("   ", "  ".join(self.classes))

        # Print each row with corresponding class label
        for i, row in enumerate(self.confusion_matrix):
            print(f"{self.classes[i]}: {row}")

    def query_matrix(self):
        """
        Generate TP, TN, FP, and FN scores for each class
        """
        truth_values = {}

        for class_name in self.classes: # iterate over each class
            truth_values[class_name] = {} # initialize empty dictionary

            class_index = self.label_to_index[class_name] # use our label map to retrieve the index corresponding to the current class

            TP = self.confusion_matrix[class_index][class_index] # TP always exist along the diagonal

            # FP = all values in the same row as TP (minus TP), that is all times another class was incorrectly predicted as this class
            FP = 0
            row = self.confusion_matrix[class_index]
            for value in row:
                FP += value
            FP -= TP

            # FN = all values in the same column as TP (minus TP), that is all times the class was incorrectly predicted as another class
            FN = 0
            for row in self.confusion_matrix:
                FN += row[class_index]
            FN -= TP

            # TN = all values in confusion matrix that isn't TP, FN, or FP
            TN = 0
            for row in self.confusion_matrix:
                for value in row:
                    TN += value
            TN -= FP
            TN -= FN
            TN -= TP

            truth_values[class_name]['TP'] = TP
            truth_values[class_name]['TN'] = TN
            truth_values[class_name]['FP'] = FP
            truth_values[class_name]['FN'] = FN

        return truth_values

    def generate_scores(self):
        """
        Use the truth values derived from the confusion matrix to calculate various scores
        """
        scores = {}
        for class_name in self.classes: # iterate over each class
            scores[class_name] = {}

            # extract truth values
            TP = self.truth_values[class_name]['TP']
            TN = self.truth_values[class_name]['TN']
            FP = self.truth_values[class_name]['FP']
            FN = self.truth_values[class_name]['FN']

            # perform calculations
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            # save and return values
            scores[class_name]['Accuracy'] = accuracy
            scores[class_name]['Precision'] = precision
            scores[class_name]['Recall'] = recall

        return scores
