from data import *

def main():

    classification = ["../data/raw_data/breast-cancer-wisconsin.data", "../data/raw_data/glass.data", "../data/raw_data/soybean-small.data"]
    regression = ["../data/raw_data/abalone.data", "../data/raw_data/forestfires.data", "../data/raw_data/machine.data"]

    for file in classification:
        data = Data(file, "class")

    for file in regression:
        data = Data(file, "regress")


main()
