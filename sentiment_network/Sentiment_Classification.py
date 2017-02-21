#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: Sentiment_Classification.py

from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import time
import sys


# Let's tweak our network from before to model these phenomena
class SentimentNetwork(object):
    def __init__(self,
                 reviews,
                 labels,
                 min_count=10,
                 polarity_cutoff=0.1,
                 hidden_nodes=10,
                 learning_rate=0.1):
        # set our random number generator
        np.random.seed(1)

        self.pre_process_data(reviews, labels, polarity_cutoff, min_count)

        self.init_network(
            len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if (labels[i] == 'positive'):
                for word in reviews[i].split(' '):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(' '):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        self.pos_neg_ratios = Counter()

        for word, cnt in list(total_counts.most_common()):
            if (cnt >= 50):
                self.pos_neg_ratios[word] = positive_counts[word] / float(
                    negative_counts[word] + 1)

        for word, ratio in list(self.pos_neg_ratios.most_common()):
            if (ratio > 1):
                self.pos_neg_ratios[word] = np.log(ratio)
            else:
                self.pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                if (total_counts[word] > min_count):
                    if (word in self.pos_neg_ratios.keys()):
                        if ((self.pos_neg_ratios[word] >= polarity_cutoff) or
                            (self.pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes,
                     learning_rate):
        # set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(
            0.0, self.output_nodes**-0.5,
            (self.hidden_nodes, self.output_nodes))

        self.learning_rate = learning_rate

        self.layer_0 = np.zeros((1, self.input_nodes))
        self.layer_1 = np.zeros((1, self.hidden_nodes))

    def update_input_layer(self, review):
        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(' '):
            if (word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] = 1  # reducing the noise

    def get_target_for_label(self, label):
        if (label == 'positive'):
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        # making our network more efficient
        training_reviews_indices = list()
        for review in training_reviews:
            unique_indices = set()
            for word in review.split(' '):
                if (word in self.word2index.keys()):
                    unique_indices.add(self.word2index[word])
            training_reviews_indices.append(list(unique_indices))

        assert (len(training_reviews) == len(training_labels))

        correct_so_far = 0

        start = time.time()

        for i in range(len(training_reviews)):
            review_indices = training_reviews_indices[i]
            label = training_labels[i]

            # Implement the forward pass here #

            # input layer
            # self.update_input_layer(review)

            # hidden layer
            # layer_1 = self.layer_0.dot(self.weights_0_1)
            self.layer_1 *= 0
            for index in review_indices:
                self.layer_1 += self.weights_0_1[index]

            # output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            # Implement the backward pass here #

            # output layer error (the difference between desired target and actual output)
            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(
                layer_2)

            # backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error  # hidden layer gradients - no nonlinearity so it's the same as the error

            # update the weights
            self.weights_1_2 -= self.learning_rate * self.layer_1.T.dot(
                layer_2_delta)
            # self.weights_0_1 -= self.learning_rate * self.layer_0.T.dot(layer_1_delta)
            for index in review_indices:
                self.weights_0_1[index] -= self.learning_rate * layer_1_delta[0]

            if (layer_2 >= 0.5 and label == 'positive'):
                correct_so_far += 1
            if (layer_2 < 0.5 and label == 'negative'):
                correct_so_far += 1

            reviews_per_second = i / float(time.time() - start)

            sys.stdout.write("\rProgress:" + str(100 * i / float(
                len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(
                    reviews_per_second)[0:5] + " #Correct:" + str(
                        correct_so_far) + " #Trained:" + str(
                            i + 1) + " Training Accuracy:" + str(
                                correct_so_far * 100 / float(i + 1))[:4] + "%")
            if (i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        correct_so_far = 0

        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if (pred == testing_labels[i]):
                correct_so_far += 1

            reviews_per_second = i / float(time.time() - start)

            sys.stdout.write("\rProgress:" + str(100 * i / float(
                len(testing_reviews)))[:4] + "% Speed(reviews/sec):" + str(
                    reviews_per_second)[0:5] + "% #Correct:" + str(
                        correct_so_far) + " #Tested:" + str(
                            i + 1) + " Testing Accuracy:" + str(
                                correct_so_far * 100 / float(i + 1))[:4] + "%")

    def run(self, review):
        # input layer
        # self.update_input_layer(review.lower())

        # hidden layer
        # layer_1 = self.layer_0.dot(self.weights_0_1)
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.split(' '):
            if (word in self.word2index.keys()):
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]

        # output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if (layer_2[0] >= 0.5):
            return 'positive'
        else:
            return 'negative'


def curate_a_dataset():
    print("------ curate a dataset start ------")
    with open('reviews.txt', 'r') as g:
        reviews = list(map(lambda x: x[:-1], g.readlines()))
    with open('labels.txt', 'r') as g:
        labels = list(map(lambda x: x[:-1], g.readlines()))
    print('returns[0]:', reviews[0], '\n')
    print('labels[0]:', labels[0])
    print("------ curate a dataset end ------\n\n")
    return reviews, labels


def quick_theory_validation(reviews, labels):
    print("------ quick theory validation start ------")
    positive_counts = Counter()
    negative_counts = Counter()
    total_counts = Counter()

    for i in range(len(reviews)):
        if (labels[i] == 'positive'):
            for word in reviews[i].split(' '):
                positive_counts[word] += 1
                total_counts[word] += 1
        else:
            for word in reviews[i].split(' '):
                negative_counts[word] += 1
                total_counts[word] += 1

    print(positive_counts.most_common()[:10], '\n')

    pos_neg_ratios = Counter()

    for word, cnt in list(total_counts.most_common()):
        if (cnt > 50):
            pos_neg_ratios[word] = positive_counts[word] / float(
                negative_counts[word] + 1)

    for word, ratio in list(pos_neg_ratios.most_common()):
        if (ratio > 1):
            pos_neg_ratios[word] = np.log(ratio)
        else:
            pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

    print(pos_neg_ratios.most_common()[:10], '\n')
    print(list(reversed(pos_neg_ratios.most_common()))[:10], '\n')

    frequency_frequency = Counter()

    for word, cnt in total_counts.most_common():
        frequency_frequency[cnt] += 1
    print(frequency_frequency.most_common()[:10], '\n')

    plt.hist(
        list(map(lambda x: x[1], pos_neg_ratios.most_common())),
        bins=100,
        normed=True)
    plt.title('Word Positive/Negative Affinity Distribution')
    plt.grid(True)
    plt.show()

    plt.hist(
        list(map(lambda x: x[1], frequency_frequency.most_common())),
        bins=100,
        normed=True)
    plt.title('The frequency distribution of the words in our corpus')
    plt.show()

    print("------ quick theory validation end ------\n\n")


def train_network(reviews, labels):
    print("------ train network start ------")
    mlp = SentimentNetwork(
        reviews[:-1000],
        labels[:-1000],
        # min_count=100,
        min_count=0,
        # polarity_cutoff=0.5,
        polarity_cutoff=0,
        learning_rate=0.01)
    mlp.test(reviews[-1000:], labels[-1000:])
    print('\n')
    mlp.train(reviews[:-1000], labels[:-1000])
    print('\n')
    # evaluate our model before training (just to show how horrible it is)
    mlp.test(reviews[-1000:], labels[-1000:])
    print()
    print("------ train network end ------\n\n")
    return mlp


def get_most_similar_words(mlp, focus='horrible'):
    most_similar = Counter()
    for word in mlp.word2index.keys():
        most_similar[word] = np.dot(mlp.weights_0_1[mlp.word2index[word]],
                                    mlp.weights_0_1[mlp.word2index[focus]])
    return most_similar


def analysis_weights(mlp):
    print("------ analysis weights start ------")
    most_similar = get_most_similar_words(mlp, focus='excellent')
    print('similar with excellent:', most_similar.most_common()[:10], '\n')
    most_similar = get_most_similar_words(mlp, focus='terrible')
    print('similar with terrible:', most_similar.most_common()[:10])

    words_to_visualize = list()
    for word, ratio in mlp.pos_neg_ratios.most_common(500):
        if (word in mlp.word2index.keys()):
            words_to_visualize.append(word)
    for word, ratio in list(reversed(mlp.pos_neg_ratios.most_common()))[0:500]:
        if (word in mlp.word2index.keys()):
            words_to_visualize.append(word)

    colors_list = list()
    vectors_list = list()
    for word in words_to_visualize:
        if (word in mlp.pos_neg_ratios.keys()):
            vectors_list.append(mlp.weights_0_1[mlp.word2index[word]])
            # green indicates positive words, black indicates negative words
            if (mlp.pos_neg_ratios[word] > 0):
                colors_list.append("#00ff00")
            else:
                colors_list.append("#000000")

    tsne = TSNE(n_components=2, random_state=0)
    words_top_ted_tsne = tsne.fit_transform(vectors_list)

    x1 = words_top_ted_tsne[:, 0]
    x2 = words_top_ted_tsne[:, 1]
    plt.scatter(x1, x2, c=colors_list)
    plt.title('vector T-SNE for most polarized words')
    for label, x, y in zip(words_to_visualize, x1, x2):
        plt.annotate(label, xy=(x, y), textcoords='offset points')
    plt.show()
    print("------ analysis weights end ------\n\n")


def main():
    reviews, labels = curate_a_dataset()
    quick_theory_validation(reviews, labels)
    mlp = train_network(reviews, labels)
    analysis_weights(mlp)


if __name__ == '__main__':
    main()
