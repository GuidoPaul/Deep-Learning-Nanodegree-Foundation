#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: nn.py

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import Input, Linear, Sigmoid, MSE, topological_sort, forward_pass, forward_and_backward, sgd_update

# ----------------------------------------------
'''
x, y, z = Input(), Input(), Input()

add = Add(x, y, z)
mul = Mul(x, y, z)

feed_dict = {x: 10, y: 20, z: 5}

graph = topological_sort(feed_dict=feed_dict)
output1 = forward_pass(add, graph)
output2 = forward_pass(mul, graph)

print("{} + {} + {} = {} (according to miniflow)".format(
    feed_dict[x], feed_dict[y], feed_dict[z], output1))

print("{} * {} * {} = {} (according to miniflow)".format(
    feed_dict[x], feed_dict[y], feed_dict[z], output2))
'''

# ----------------------------------------------

X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
forward_pass(graph)
print(g.value)

# ----------------------------------------------

y, a = Input(), Input()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}

graph = topological_sort(feed_dict)
forward_pass(graph)
print(cost.value)

# ----------------------------------------------

X, W, b = Input(), Input(), Input()
y = Input()
f = Linear(X, W, b)
a = Sigmoid(f)
cost = MSE(y, a)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2.], [3.]])
b_ = np.array([-3.])
y_ = np.array([1, 2])

feed_dict = {
    X: X_,
    y: y_,
    W: W_,
    b: b_,
}

graph = topological_sort(feed_dict)
forward_and_backward(graph)
# return the gradients for each Input
gradients = [t.gradients[t] for t in [X, y, W, b]]

print(gradients)

# ----------------------------------------------

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {X: X_, y: y_, W1: W1_, b1: b1_, W2: W2_, b2: b2_}

epochs = 1000
# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
