class Data:
    def __init__(self):
        self.num_classes = 0
        self.classes = []
        self.num_features = 0
        self.num_entries = 0
        self.data = []

    def process_file(self, path):
        with open(path, "r") as file:
            num_lines = 0
            num_features = 0
            classes = []
            for line in file:
                num_lines += 1
                split = line.strip('\n').split(",")
                if len(split) > 1:
                    num_features = len(split) - 1
                    class_name = split[num_features]
                    if not class_name in classes:
                        classes.append(class_name)
            num_classes = len(classes)

            print("num classes: ", num_classes)
            print("classes: " , classes)
            print("num features: ", num_features)
            print("num lines: ", num_lines)

        file.close()

        with open(path, "r") as file:
            averages = [0] * num_features
            data = [[0] * num_features for _ in range(num_lines)]
            line_count = 0
            for line in file:
                split = line.strip('\n').split(",")
                if len(split) > 1:
                    for i in range(num_features):
                        if not split[i] == '?':
                            averages[i] += float(split[i])
                            data[line_count][i] += float(split[i])

                line_count += 1

            print(data)

            for i in range(num_features):
                averages[i] /= num_lines
                averages[i] = round(averages[i], 2)

            print("averages: ", averages)
            print("\n")

            self.num_classes = num_classes
            self.classes = classes
            self.num_features = num_features
            self.num_entries = num_lines
            self.data = data

        file.close()

breast_cancer = Data()
breast_cancer.process_file("../data/breast-cancer-wisconsin.data")

iris = Data()
iris.process_file("../data/iris.data")

glass = Data()
glass.process_file("../data/glass.data")

soybean = Data()
soybean.process_file("../data/soybean-small.data")
