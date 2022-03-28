# Neural Network class

import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, layer1_nodes, layer2_nodes, learning_rate):
        self.model = MLPClassifier(hidden_layer_sizes=(layer1_nodes, layer2_nodes), activation='relu',
                                   solver='sgd', alpha=0.01, batch_size=200, learning_rate='constant',
                                   learning_rate_init=learning_rate, max_iter=100, tol=1e-4,
                                   early_stopping=False, validation_fraction=0.1, momentum=0.5,
                                   n_iter_no_change=100, random_state=42)

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)  # predict on test data

        print('\nEvaluate on the Test Set')
        print(classification_report(y_test, predictions))  # produce classification report
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, predictions))  # produce confusion matrix

    def fit(self, x_train, y_train):
        # Fit the model and report training time
        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()
        print('\nTraining Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, x):
        start_time = time.time()
        predictions = self.model.predict(x)
        end_time = time.time()
        print('\nTesting Set: {:.4f} seconds'.format(end_time-start_time))

        return predictions

    def experiment(self, x_train, x_test, y_train, y_test):
        self.fit(x_train, y_train)
        self.evaluate(x_test, y_test)
