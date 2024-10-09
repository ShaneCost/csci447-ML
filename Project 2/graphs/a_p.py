import pandas as pd

class DecisionMatrix:
    def __init__(self, confusion_matrix_str):
        self.confusion_matrix, self.classes = self.parse_confusion_matrix(confusion_matrix_str)

    def parse_confusion_matrix(self, matrix_str):
        # Split the input string into lines and parse it
        lines = matrix_str.strip().split('\n')
        classes = []
        matrix = []
        
        for line in lines[2:]:  # Skip the first two lines (labels)
            parts = line.split()
            classes.append(parts[0].strip(':'))  # Store the class label (2 or 4)
            row = list(map(int, parts[1:]))  # Convert remaining parts to integers
            matrix.append(row)
        
        return matrix, classes

    def calculate_metrics(self, class_index):
        true_positive = self.confusion_matrix[class_index][class_index]
        false_positive = sum(self.confusion_matrix[i][class_index] for i in range(len(self.confusion_matrix))) - true_positive
        false_negative = sum(self.confusion_matrix[class_index]) - true_positive
        true_negative = sum(sum(row) for row in self.confusion_matrix) - (true_positive + false_positive + false_negative)

        accuracy = (true_positive + true_negative) / sum(sum(row) for row in self.confusion_matrix)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        return {
            'Class': self.classes[class_index],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

    def display_metrics(self):
        all_metrics = []
        for i in range(len(self.classes)):
            metrics = self.calculate_metrics(i)
            all_metrics.append(metrics)
        
        df = pd.DataFrame(all_metrics)
        print(df)


if __name__ == "__main__":
    # Input confusion matrix as a string
    confusion_matrix_str = """
    Classes: 1:   2:   3:   5:   6:   7:
    1:       60    0    0    0    0    0
    2:       4     55   9    1    1    0
    3:       0     2    13   0    0    0
    5:       0     7    0    2    0    3
    6:       0     2    0    0    5    2
    7:       0     0    0    1    4    22
    """
    
    matrix = DecisionMatrix(confusion_matrix_str)
    matrix.display_metrics()
