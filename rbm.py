#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RBMLayer(object):
    def __init__(self, visible_size, hidden_size):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.W = self.initialize_weights()
        self.b = self.initialize_visible_bias()
        self.c = self.initialize_hidden_bias()

    def initialize_weights(self):
        return np.random.normal(scale=0.01,
                                size=(self.visible_size, self.hidden_size))

    def initialize_visible_bias(self):
        return np.zeros(self.visible_size)

    def initialize_hidden_bias(self):
        return np.zeros(self.hidden_size)

    # 输入观测变量v, 输出隐变量的值h
    def forward(self, v):
        return sigmoid(np.dot(v, self.W) + self.c)

    # 输入隐变量的值h, 输出观测变量值y
    def backward(self, h):
        return sigmoid(np.dot(h, self.W.T) + self.b)

    # k-step contrastive divergence 训练算法
    def contrastive_divergence(self, k, data_set):
        pass
