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
        self.micro_scores = self.generate_scores()
        self.test_score = self.generate_test_score()

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

            # Error-checking for each metric calculation to avoid division by zero

            # Accuracy
            accuracy_denominator = TP + TN + FP + FN
            if accuracy_denominator != 0:
                accuracy = (TP + TN) / accuracy_denominator
            else:
                accuracy = 0.0  # Or an appropriate value for division by zero

            # Precision
            precision_denominator = TP + FP
            if precision_denominator != 0:
                precision = TP / precision_denominator
            else:
                precision = 0.0  # Or an appropriate value for division by zero

            # Recall
            recall_denominator = TP + FN
            if recall_denominator != 0:
                recall = TP / recall_denominator
            else:
                recall = 0.0  # Or an appropriate value for division by zero

            # F1 Score
            if precision + recall != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0  # Or an appropriate value for division by zero

            # Save calculated values
            scores[class_name]['Accuracy'] = accuracy
            scores[class_name]['Precision'] = precision
            scores[class_name]['Recall'] = recall
            scores[class_name]['F1'] = f1

        return scores

    def generate_test_score(self):
        f1 = 0
        num_class = 0
        for class_name in self.classes:
            num_class += 1
            f1 += self.micro_scores[class_name]['F1']

        return f1/num_class