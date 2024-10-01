import numpy as np
from collections import Counter
from data import Data

class KNN:
    def __init__(self, training_data, testing_data):
        self.training_data = np.array(training_data)
        self.testing_data = np.array(testing_data)
        self.is_classification = True
    
    def classify(self, testing_data, k, p):

        predications = []
        for test_point in testing_data:

            distances = []
            for data_point in self.training_data:
                distance = self.get_distance(test_point[:-1], data_point[:-1], p)
                distances.append((distance, data_point[-1]))
            
            distances = sorted(distances)

            predication = self.vote(distances, k)
            predications.append(predication)

        return predications
        
    
    def vote(self, distances, k):

        # Voting / Average
        distances = distances[0:k]
        if(self.is_classification):
            classes = [t[1] for t in distances]
            prediction = Counter(classes).most_common(1)[0][0]
        else:
            target_values = [t[1] for t in distances]
            prediction = sum(target_values)/len(target_values)

        return prediction
    
    def get_actual(self):
        actual = []
        for row in self.testing_data:
            actual.append(row[-1])
        return actual

    def get_distance(self, x, y, p):

        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)   
        distance = np.sum(np.abs(x - y) ** p) ** (1 / p)
       
        return distance


def main():

    path = "Project 2\data\soybean-small.data"
    data = Data(path, "class")

    training_set = data.get_training_set(1)
    test_set = data.get_test_set(1)

    knn = KNN(training_set, test_set)
    predications = knn.classify(test_set, 2, 2)
    actual = knn.get_actual()

    print(actual)
    print(predications)

main()