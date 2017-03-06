#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: test.py

import numpy as np


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
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))

    def __forward_propagate(self, data):
        # Progapagate through network and hold values for use in back-propagation
        activation_values = {}
        activation_values[1] = data
        for layer in range(2, self.num_layers + 1):
            data = np.dot(data.T,
                          self.weights[layer - 1][:-1, :]) + self.weights[
                              layer - 1][-1, :].T  # self.biases[layer]
            data = self.__sigmoid(data).T
            activation_values[layer] = data
        return activation_values

    def __back_propagate(self, output, target):
        deltas = {}
        # Delta of output layer
        deltas[self.num_layers] = output[self.num_layers] - target

        # Delta of hidden layer
        for layer in reversed(
                range(2, self.num_layers)):  # all layers except input/output
            a_val = output[layer]
            weights = self.weights[layer][:-1, :]
            prev_deltas = deltas[layer + 1]
            deltas[layer] = np.multiply(
                np.dot(weights, prev_deltas), self.__sigmoid_derivative(a_val))

        # Calculate total adjustments based on deltas
        for layer in range(1, self.num_layers):
            self.adjustments[layer] += np.dot(deltas[layer + 1],
                                              output[layer].T).T

    def __gradient_descent(self, batch_size, learning_rate):
        # Calculate partial derivative and take a step in that direction
        for layer in range(1, self.num_layers):
            partial_d = (1 / batch_size) * self.adjustments[layer]  # ?
            self.weights[layer][:-1, :] += learning_rate * -partial_d
            self.weights[layer][-1, :] += learning_rate * 1e-3 * -partial_d[
                -1, :]  # ?

    def add_input_layer(self, n_units):
        self.n_units[0] = n_units
        self.num_layers = 1

    def add_layer(self, n_units):
        # Create weights with n_units specified + biases
        prev_n_units = self.n_units[self.num_layers - 1]
        self.weights[self.num_layers] = 2 * np.random.random(
            (prev_n_units + 1, n_units)) - 1  # prev_n_units + 1 biases
        # Initialize the adjustements for these weights to zero
        self.adjustments[self.num_layers] = np.zeros((prev_n_units, n_units))
        self.n_units[self.num_layers] = n_units
        self.num_layers += 1

    def train(self,
              inputs,
              targets,
              num_epochs,
              learning_rate=0.1,
              stop_accuracy=1e-5):
        error = []
        for iteration in range(num_epochs):
            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]
                # Pass the training set through our nenral network
                output = self.__forward_propagate(x)

                # Calculate the error
                loss = self.__sum_squared_error(output[self.num_layers], y)
                error.append(loss)

                # Calculate Adjustments
                self.__back_propagate(output, y)

            self.__gradient_descent(i, learning_rate)

            # Check if accuarcy criterion is satisfied
            if np.mean(error[-(i + 1):]) < stop_accuracy and iteration > 0:
                break

        return (np.asarray(error), iteration + 1)

    def predict(self, data):
        outputs = []
        for i in range(len(data)):
            x = data[i]
            output = self.__forward_propagate(x)
            outputs.append(output[self.num_layers])
        return outputs

    def test():
        pass


def main():
    # Create instance of a neural network
    nn = NeuralNetwork()

    # Add layers (input layer is created by default)
    nn.add_input_layer(2)
    nn.add_layer(9)
    nn.add_layer(1)

    # XOR function
    training_data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2,
                                                                         1)
    training_labels = np.asarray([[0], [1], [1], [0]])

    testing_data = np.asarray([[0, 0.1], [0.9, 0.9], [1.1, 0.2]]).reshape(3, 2,
                                                                          1)

    error, iteration = nn.train(training_data, training_labels, 5000)
    print('Error = ', np.mean(error[-4:]))
    print('Epoches needed to train = ', iteration)

    print((np.array(nn.predict(testing_data))[:, 0, 0] >= 0.5).astype(np.int))


if __name__ == '__main__':
    main()
