#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: Sentiment_Analysis_with_TFLearn.py

from collections import Counter
import pandas as pd
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical


def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if (idx is None):
            continue
        else:
            word_vector[word2idx[word]] += 1
    return np.array(word_vector)


# Preparing the data
# read the data
reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

# counting word frequency
total_counts = Counter()

for idx, review in reviews.iterrows():
    total_counts.update(review[0].split(' '))

print('Total words in data set:', len(total_counts))

vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])

# create the word-to-index dictionary here
word2idx = {word: i for i, word in enumerate(vocab)}

word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int)

for idx, review in reviews.iterrows():
    word_vectors[idx] = text_to_vector(review[0])

print(word_vectors[:5, :23])

# train, validation, test sets
Y = (labels == 'positive').astype(np.int)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records * test_fraction)], shuffle[int(
    records * test_fraction):]
trainX, trainY = word_vectors[train_split, :], to_categorical(
    Y.values[train_split], 2)
testX, testY = word_vectors[test_split, :], to_categorical(
    Y.values[test_split], 2)


# Building the network
def build_model():
    # Inputs
    net = tflearn.input_data([None, len(vocab)])

    # Hidden layer(s)
    net = tflearn.fully_connected(net, 400, activation='ReLU')
    net = tflearn.fully_connected(net, 100, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')

    # Output layer
    net = tflearn.fully_connected(net, 2, activation='softmax')

    # train network
    net = tflearn.regression(
        net,
        optimizer='sgd',
        loss='categorical_crossentropy',
        learning_rate=0.1)

    model = tflearn.DNN(net)
    return model


model = build_model()

# Training the network
model.fit(trainX,
          trainY,
          validation_set=0.1,
          show_metric=True,
          batch_size=128,
          n_epoch=50)

# Testing
predictions = (np.array(model.predict(testX))[:, 0] >= 0.5).astype(np.int)
test_accuracy = np.mean(predictions == testY[:, 0], axis=0)
print('Test accuracy:', test_accuracy)


# Try out your own text!
def test_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    print('Sentence: {}'.format(sentence))
    print('P(positive) = {:.3f} :'.format(positive_prob), 'Positive'
          if positive_prob > 0.5 else 'Negative')


sentence = "Moonlight is by far the best movie of 2016."
test_sentence(sentence)
sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
test_sentence(sentence)
