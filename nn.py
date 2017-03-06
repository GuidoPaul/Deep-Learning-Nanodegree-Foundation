#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: nn.py

from miniflow import Input, Add, Mul, topological_sort, forward_pass

x, y, z = Input(), Input(), Input()

add = Add(x, y, z)
mul = Mul(x, y, z)

feed_dict = {x: 10, y: 20, z: 5}

sorted_nodes = topological_sort(feed_dict=feed_dict)
output1 = forward_pass(add, sorted_nodes)
output2 = forward_pass(mul, sorted_nodes)

print("{} + {} + {} = {} (according to miniflow)".format(
    feed_dict[x], feed_dict[y], feed_dict[z], output1))

print("{} * {} * {} = {} (according to miniflow)".format(
    feed_dict[x], feed_dict[y], feed_dict[z], output2))
