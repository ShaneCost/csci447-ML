import numpy as np
from collections import Counter
from data import Data

class KNN:
    def __init__(self, training_data, testing_data):
        self.training_data = np.array(training_data)
        self.testing_data = np.array(testing_data)
        self.is_classification = True
    
    def knn(self, test_point, k, p):
        distances = []
        for data_point in self.training_data:

            distance = self.get_distance(test_point[:-1], data_point[:-1], p)
            distances.append((distance, data_point[-1]))
        
        distances = sorted(distances)
        
        #Voting
        distances = distances[0:k]
        classes = [t[1] for t in distances]
        prediction = Counter(classes).most_common(1)[0][0]

        print(prediction)

    def get_distance(self, x, y, p):

        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)   
        distance = np.sum(np.abs(x - y) ** p) ** (1 / p)
       
        return distance



def main():

    path = "Project 2\data\\raw_data\soybean-small.data"
    data = Data(path, "class")


    training_set = data.get_training_set(1)
    test_set = data.get_test_set(1)


    knn = KNN(training_set, test_set)
    knn.knn(test_set[0], 2, 2)

main()