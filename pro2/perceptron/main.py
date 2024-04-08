import csv
import random
import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate):
        self.weights = np.random.rand(num_features)
        self.bias = random.random()
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, iterations):
        for _ in range(iterations):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        num_features = len(next(csv_reader)) - 1
        file.seek(0)
        for row in csv_reader:
            label = row[-1]
            features = [float(x) for x in row[:-1]]
            dataset.append(features + [label])
    return dataset, num_features

def main():

    training_data, num_features = load_dataset('data/perceptron.data')
    test_data, _ = load_dataset('data/perceptron.test.data')
    while True:

        learning_rate = float(input("Enter the learning rate (e.g., 0.01) or type '0' to quit: "))
        if learning_rate == 0:
            break
        iterations = int(input("Enter the number of iterations: "))

        training_inputs = np.array([data[:-1] for data in training_data])
        labels = np.array([1 if data[-1] == 'Iris-versicolor' else 0 for data in training_data])

        perceptron = Perceptron(num_features=num_features, learning_rate=learning_rate)
        perceptron.train(training_inputs, labels, iterations=iterations)

        correct_predictions = 0
        for data in test_data:
            inputs = np.array(data[:-1])
            true_label = 1 if data[-1] == 'Iris-versicolor' else 0
            prediction = perceptron.predict(inputs)
            if prediction == true_label:
                correct_predictions += 1
        accuracy = correct_predictions / len(test_data) * 100
        print(f"Accuracy on test set for learning rate {learning_rate} and {iterations} iterations: {accuracy:.2f}%")

        manual_input = input("Enter comma-separated values for manual classification (sepal length, sepal width, petal length, petal width): ")
        if manual_input.lower() == 'e':
            break
        inputs = np.array([float(x) for x in manual_input.split(',')])
        prediction = perceptron.predict(inputs)
        if prediction == 1:
            print("Prediction: Iris-versicolor")
        else:
            print("Prediction: Iris-virginica")

if __name__ == "__main__":
    main()
