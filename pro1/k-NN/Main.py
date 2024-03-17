import csv
import numpy as np
import os

def main(k, trSet, tstSet):
    v_train = readVec(trSet)
    cl_train = readClass(trSet)
    v_test = readVec(tstSet)
    cl_test = readClass(tstSet)

    from KNNClassifier import KNNClassifier
    kNN = KNNClassifier(k ,v_train ,cl_train)

    results = kNN.evaluate(v_test)

    correct = sum(1 for pred, true in zip(results, cl_test) if pred == true)
    total_samples = len(cl_test)
    accuracy = correct / total_samples

    print("Classified with accuracy:",accuracy * 100, "%")

    while True:
        usrVector = input("Enter a custom vector to classify as comma - separated values"
                          "\t(press q to quit): ")
        if usrVector == 'q':
            break
        usrVector = np.array([float(val) for val in usrVector.split(',')])
        result = kNN.evaluate([usrVector])
        print("Evaluated class:", result[0])


def readVec(fileName):
    read = []
    path = os.path.join("data", fileName)
    with open(path, 'r') as file:
        rd = csv.reader(file)
        for row in rd:
            read.append([float(val) for val in row[:-1]])
    return np.array(read)


def readClass(fileName):
    classes = []
    path = os.path.join("data", fileName)
    with open(path, 'r') as file:
        rd = csv.reader(file)
        for row in rd:
            classes.append(row[-1])
    return classes


if __name__ == "__main__":
    k = int(input("Enter the value of k:"))
    trainingSet = input("Enter the name of a file with training data:")
    testSet = input("Enter the name of a file with test data:")
    main(k,trainingSet,testSet)