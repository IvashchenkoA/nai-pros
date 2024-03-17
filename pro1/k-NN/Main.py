import csv
import numpy as np

if __name__ == "__main__":
    k = int(input("Enter tha value of k:"))
    trainingSet = input("Enter the name of a file with training data:")
    testSet = input("Enter the name of a file with test data:")


def readFile(fileName):
    read = []
    with open(fileName, 'r') as file:
        rd = csv.reader(file)
        for row in rd:
            read.append([float(val) for val in row])
    return np.array(read)

