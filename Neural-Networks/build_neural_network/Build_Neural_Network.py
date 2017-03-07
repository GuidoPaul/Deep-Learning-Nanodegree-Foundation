#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: test.py

import tflearn.datasets.mnist as mnist
import numpy as np
import sys
import time


class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)  # Seed the random number generator
        self.n_units = {}  # Set the number of nodes per layer
        self.weights = {}  # Create dict to hold weights
        self.num_layers = 0  # Set initial number of layer to one (input layer)
        self.adjustments = {}  # Create dict to hold adjustments

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __sum_squared_error(self, outputs, targets):
        return 0.5 * np.mean((outputs - targets)**2)

    def __forward_propagate(self, data):
        # Progapagate through network and hold values for use in back-propagation
        a_val = {}
        a_val[1] = data
        for layer in range(2, self.num_layers + 1):
            z_val = np.dot(self.weights[layer - 1], a_val[layer - 1])
            a_val[layer] = self.__sigmoid(z_val)
        return a_val

    def __back_propagate(self, outputs, target, learning_rate):
        deltas = {}
        # Delta of output layer
        train_output = outputs[self.num_layers]
        deltas[self.num_layers] = (
            train_output - target) * self.__sigmoid_derivative(train_output)

        # Delta of hidden layer
        for layer in reversed(
                range(2, self.num_layers)):  # all layers except input/output
            prev_deltas = deltas[layer + 1]
            deltas[layer] = np.dot(
                self.weights[layer].T,
                prev_deltas) * self.__sigmoid_derivative(outputs[layer])

        # Calculate adjustments based on deltas
        for layer in range(1, self.num_layers):
            self.adjustments[layer] = np.dot(deltas[layer + 1],
                                             outputs[layer].T)

    def __gradient_descent(self, batch_size, learning_rate):
        # Calculate partial derivative and take a step in that direction
        for layer in range(1, self.num_layers):
            self.weights[layer] -= learning_rate * self.adjustments[layer]

    def add_input_layer(self, n_units):
        self.n_units[0] = n_units
        self.num_layers = 1

    def add_layer(self, n_units):
        # Create weights with n_units specified + biases
        prev_n_units = self.n_units[self.num_layers - 1]
        self.weights[self.num_layers] = np.random.normal(
            0.0, self.n_units[0]**-0.5, (n_units, prev_n_units))
        # Initialize the adjustements for these weights to zero
        self.adjustments[self.num_layers] = np.zeros((n_units, prev_n_units))
        self.n_units[self.num_layers] = n_units
        self.num_layers += 1

    def train(self,
              inputs,
              targets,
              num_epochs,
              learning_rate=0.1,
              stop_accuracy=1e-5):

        error = []
        start = time.time()

        for iteration in range(num_epochs):

            correct_so_far = 0

            for i in range(len(inputs)):
                x = np.asarray(inputs[i]).reshape(784, 1)
                y = np.asarray(targets[i]).reshape(10, 1)

                # Pass the training set through our nenral network
                outputs = self.__forward_propagate(x)

                # Calculate the error
                loss = self.__sum_squared_error(outputs[self.num_layers], y)
                error.append(loss)

                # Calculate Adjustments
                self.__back_propagate(outputs, y, learning_rate)

                # Adjust Weights
                self.__gradient_descent(i, learning_rate)

                if (targets[i].argmax() == outputs[self.num_layers].flatten()
                        .argmax()):
                    correct_so_far += 1

                data_per_second = i / float(time.time() - start)

                sys.stdout.write("\rProgress:" + ('%.3f' % (100 * i / float(
                    len(inputs)))) + "% Speed(data/sec):" + (
                        '%.3f' % data_per_second
                    ) + " Lost:" + ('%.3f' % loss) + " #Correct:" + str(
                        correct_so_far) + " #Trained:" + str(
                            i + 1) + " Training Accuracy:" + ('%.3f' % (
                                correct_so_far * 100 / float(i + 1))) + "%")

                if (i % 2500 == 0):
                    print("")

            print("\nEpoch: {}, Lost: {}".format(iteration + 1,
                                                 np.mean(error[-4:])))

            # Check if accuarcy criterion is satisfied
            if np.mean(error[-(i + 1):]) < stop_accuracy and iteration > 0:
                break

        return (np.asarray(error), iteration + 1)

    def predict(self, data):
        outputs = []
        for i in range(len(data)):
            x = np.asarray(data[i]).reshape(784, 1)
            output = self.__forward_propagate(x)
            outputs.append(output[self.num_layers])
        return outputs

    def test():
        pass


def build_nn(n_units):
    # Create instance of a neural network
    nn = NeuralNetwork()

    # Add layers (input layer is created by default)
    nn.add_input_layer(n_units)
    nn.add_layer(400)
    nn.add_layer(32)
    nn.add_layer(10)

    return nn


def main():
    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

    nn = build_nn(trainX.shape[1])

    error, iteration = nn.train(trainX, trainY, 2)
    print('Epoch: {}, Lost: {}'.format(iteration, np.mean(error[-4:])))

    preds = np.array(nn.predict(testX)).argmax(axis=1).flatten()
    actual = testY.argmax(axis=1)
    test_accuracy = np.mean(preds == actual, axis=0)

    print("Test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    main()
