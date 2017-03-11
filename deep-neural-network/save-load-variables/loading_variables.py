#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: loading_variables.py

import tensorflow as tf

# The file path to save the data
save_file = './model.ckpt'

# Remove the previous weights and bias
tf.reset_default_graph()

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
biases = tf.Variable(tf.truncated_normal([3]), name='bias_0')

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, save_file)

    # Show the values of weights and bias
    print('Weights:')
    print(sess.run(weights))
    print('Biases:')
    print(sess.run(biases))
