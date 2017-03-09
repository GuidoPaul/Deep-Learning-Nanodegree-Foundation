#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: gd.py

# import f


def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    """
    # TODO: Implement gradient descent.

    # Return the new value for x
    x = x - learning_rate * gradx
    return x
