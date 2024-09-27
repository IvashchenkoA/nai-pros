import csv
import numpy as np

def preprocess_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            text = row[1].lower()
            texts.append(text)
            labels.append(row[0])
    return texts, labels

def count_occurrences(text):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    counts = [0] * 26
    for char in text:
        if char in alphabet:
            index = alphabet.index(char)
            counts[index] += 1
    return counts

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def generate_input_vectors(texts):
    input_vectors = []
    for text in texts:
        counts = count_occurrences(text)
        normalized = normalize_vector(counts)
        input_vectors.append(normalized)
    return np.array(input_vectors)

class SingleLayerNetwork:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)

    def train(self, X, y, learning_rate=0.01, epochs=100):
        n_samples = X.shape[0]
        one_hot_y = np.zeros((n_samples, self.weights.shape[1]))
        for i in range(n_samples):
            one_hot_y[i, y[i]] = 1
        for _ in range(epochs):
            predictions = self.predict(X)
            error = one_hot_y - predictions
            self.weights += learning_rate * np.dot(X.T, error)

    def predict(self, X):
        return np.dot(X, self.weights)

def calc_accuracy(model, X_test, y_test, label_map):
    predictions = model.predict(X_test)
    correct = 0
    total = len(y_test)
    misclassified_texts = []

    for i in range(total):
        predicted_label = np.argmax(predictions[i])
        true_label = y_test[i]

        if predicted_label == true_label:
            correct += 1
        else:
            misclassified_texts.append((test_texts[i], list(label_map.keys())[true_label]))

    accuracy = (correct / total) * 100
    return accuracy, misclassified_texts

def classify_text(text, model, label_map):
    input_vector = generate_input_vectors([text])
    prediction = model.predict(input_vector)
    language_index = np.argmax(prediction)
    languages = list(label_map.keys())
    return languages[language_index]


if __name__ == "__main__":
    train_texts, train_labels = preprocess_data('lang.train.csv')
    test_texts, test_labels = preprocess_data('lang.test.csv')

    X_train = generate_input_vectors(train_texts)
    X_test = generate_input_vectors(test_texts)

    unique_labels = set(train_labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_map[label] for label in train_labels])
    y_test = np.array([label_map[label] for label in test_labels])

    input_size = 26
    output_size = len(label_map)
    model = SingleLayerNetwork(input_size, output_size)
    model.train(X_train, y_train)

    accuracy, misclassified_texts = calc_accuracy(model, X_test, y_test, label_map)
    print("Test accuracy:", accuracy)

    print("\nMisclassified Texts:")
    for text, true_label in misclassified_texts:
        print("Text:", text)
        print("True Label:", true_label)
        print()

    while True:
        text = input("Enter a text to classify (type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        language = classify_text(text, model, label_map)
        print("Predicted language:", language)
