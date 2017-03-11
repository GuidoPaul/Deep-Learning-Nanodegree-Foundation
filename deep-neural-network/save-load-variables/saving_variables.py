#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: saving_variables.py

import tensorflow as tf

# The file path to save the data
save_file = './model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
biases = tf.Variable(tf.truncated_normal([3]), name='bias_0')

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize all the Variables
    sess.run(tf.global_variables_initializer())

    # Show the values of weights and bias
    print('Weights:')
    print(sess.run(weights))
    print('Biases:')
    print(sess.run(biases))

    # Save the model
    saver.save(sess, save_file)
