#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: convolve.py

import numpy as np
import cv2


def convolve(img, kernel, mode='valid', stride=1, correlate=False):
    if (not correlate):
        kernel = kernel[::-1, ::-1]

    # compute output image size
    iH, iW = img.shape
    kH, kW = kernel.shape

    if (mode == 'valid'):
        pH, pW = 0, 0
    elif (mode == 'same'):
        pH, pW = int((kH - 1) / 2), int((kW - 1) / 2)

    extH, extW = iH + 2 * pH, iW + 2 * pW
    extImage = np.zeros((extH, extW))
    extImage[pH:pH + iH, pW:pW + iW] = img

    oH = int((extH - kH) / stride + 1)
    oW = int((extW - kW) / stride + 1)

    output = np.zeros((oH, oW))

    # loop over every pixel, compute the dot product
    for y in np.arange(oH):
        for x in np.arange(oW):
            # range of interest
            y_start, x_start = y * stride, x * stride
            y_end, x_end = y_start + kH, x_start + kW
            roi = extImage[y_start:y_end, x_start:x_end]
            output[y, x] = np.sum(roi * kernel)

    return output


def normalize(img):
    mini = np.min(img)
    maxi = np.max(img)

    return (img - mini) / (maxi - mini) * 255


img = cv2.imread("Lenna.png", cv2.IMREAD_GRAYSCALE)

noise = np.random.normal(0, 20, img.shape)
noise_img = img + noise
cv2.imwrite("Lenna_noise.png", noise_img)

k_average = np.ones((5, 5)) / 25
conv_average = convolve(noise_img, k_average, mode='valid', stride=2)
cv2.imwrite("Lenna_average.png", conv_average)

k_gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
conv_gaussian = convolve(img, k_gaussian)
cv2.imwrite("Lenna_gaussian.png", conv_gaussian)

k_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 16.0
conv_edge = convolve(img, k_edge)
cv2.imwrite("Lenna_edge.png", normalize(conv_edge))

k_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 16.0
conv_edge = convolve(img, k_edge, correlate=True)
cv2.imwrite("Lenna_edge_corre.png", normalize(conv_edge))

k_sharpen = np.zeros((3, 3))
k_sharpen[1, 1] = 2
k_sharpen -= np.ones((3, 3)) / 9
conv_sharpen = convolve(img, k_sharpen)
cv2.imwrite("Lenna_sharpen.png", conv_sharpen)
