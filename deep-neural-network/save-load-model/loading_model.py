#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: loading_model.py

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Remove previous Tensors and Operations
tf.reset_default_graph()

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('mnist', one_hot=True)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights and biases
weights = tf.Variable(tf.random_normal([n_input, n_classes]), name='weights_0')
biases = tf.Variable(tf.random_normal([n_classes]), name='biases_0')

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), biases)

# Define loss and optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
save_file = './train_model.ckpt'

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images,
                   labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
