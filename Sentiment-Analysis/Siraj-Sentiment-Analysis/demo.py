#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: demo.py

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np

# IMDB Dataset loading
train, valid, test = imdb.load_data(
    path='imdb.pkl', n_words=10000, valid_portion=0.1)

trainX, trainY = train
validX, validY = valid
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
validX = pad_sequences(validX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
validY = to_categorical(validY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(
    net,
    optimizer='adam',
    learning_rate=0.0001,
    loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX,
          trainY,
          validation_set=(validX, validY),
          show_metric=True,
          batch_size=32)

model.predict(testX[:200])
model.evaluate(testX, testY, batch_size=128)

# Testing
batch_size = 128
test_accuracys = []
for i in range(len(testY) // batch_size):
    batch_s = i * batch_size
    batch_e = i * batch_size + batch_size - 1
    predictions = (np.array(model.predict(testX[batch_s:batch_e]))[:, 0] >= 0.5
                   ).astype(np.int)
    test_accuracy = np.mean(
        predictions == testY[batch_s:batch_e][:, 0], axis=0)
    test_accuracys.append(test_accuracy)
print(np.mean(test_accuracys))
print(model.evaluate(testX, testY, batch_size=128))
