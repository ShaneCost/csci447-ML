def main(path):
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

    with open(path, "r") as file:
        averages = [0] * num_features
        for line in file:
            split = line.strip('\n').split(",")
            if len(split) > 1:
                for i in range(num_features):
                    if not split[i] == '?':
                        averages[i] += float(split[i])


        for i in range(num_features):
            averages[i] /= num_lines
            averages[i] = round(averages[i], 2)

        print("averages: " , averages)

main("../data/breast-cancer-wisconsin.data")
main("../data/iris.data")
