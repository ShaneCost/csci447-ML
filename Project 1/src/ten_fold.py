class TenFold:

    def __init__(self):
        self.data = []

        self.num_classes = 0
        self.classes = []
        self.num_features = 0
        self.num_entries = 0

        self.fold_1 = []
        self.fold_2 = []
        self.fold_3 = []
        self.fold_4 = []
        self.fold_5 = []
        self.fold_6 = []
        self.fold_7 = []
        self.fold_8 = []
        self.fold_9 = []
        self.fold_10 = []

    def load(self, path):
        with open(path, "r") as file:
            num_lines = 0
            num_features = 0
            classes = []
            for line in file:
                num_lines += 1
                split = line.strip('\n').split(",")
                if len(split) > 1:
                    self.data.append(split)
                    num_features = len(split) - 1
                    class_name = split[num_features]
                    if not class_name in classes:
                        classes.append(class_name)
            num_classes = len(classes)

        self.num_classes = num_classes
        self.classes = classes
        self.num_features = num_features
        self.num_entries = num_lines

        file.close()

        # Calculate number of entries per fold
        num_entries_per_fold = num_lines // 10
        extra_entries = num_lines % 10  # Remainder

        # Split the data into 10 folds
        folds = []
        start_idx = 0

        for i in range(10):
            end_idx = start_idx + num_entries_per_fold
            if i < extra_entries:  # Add one extra entry to the first 'extra_entries' folds
                end_idx += 1
            folds.append(self.data[start_idx:end_idx])
            start_idx = end_idx

        # Assign folds to class attributes
        self.fold_1 = folds[0]
        self.fold_2 = folds[1]
        self.fold_3 = folds[2]
        self.fold_4 = folds[3]
        self.fold_5 = folds[4]
        self.fold_6 = folds[5]
        self.fold_7 = folds[6]
        self.fold_8 = folds[7]
        self.fold_9 = folds[8]
        self.fold_10 = folds[9]

def main():
    breast_cancer = TenFold()
    breast_cancer.load("../data/breast-cancer-wisconsin_processed.data")

    glass = TenFold()
    glass.load("../data/glass_processed.data")

    soybean = TenFold()
    soybean.load("../data/soybean-small_processed.data")

    iris = TenFold()
    iris.load("../data/iris_processed.data")

    votes = TenFold()
    votes.load("../data/house-votes-84_processed.data")

    breast_cancer_noisy = TenFold()
    breast_cancer_noisy.load("../data/breast-cancer-wisconsin_noisy.data")

    glass_noisy = TenFold()
    glass_noisy.load("../data/glass_noisy.data")

    soybean_noisy = TenFold()
    soybean_noisy.load("../data/soybean-small_noisy.data")

    iris_noisy = TenFold()
    iris_noisy.load("../data/iris_noisy.data")

    votes_noisy = TenFold()
    votes_noisy.load("../data/house-votes-84_noisy.data")

main()