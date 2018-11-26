#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from functools import reduce


class VectorOp(object):
    """
    实现向量计算操作
    """
    @staticmethod
    def dot(x, y):
        """
        计算两个向量x和y的内积
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]按元素相乘
        # 变成[x1*y1, x2*y2, x3*y3]
        # 然后利用reduce求和
        return reduce(lambda a, b: a + b, VectorOp.element_multiply(x, y), 0.0)

    @staticmethod
    def element_multiply(x, y):
        """
        将两个向量x和y按元素相乘
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        # 变成[(x1,y1),(x2,y2),(x3,y3),...]
        # 然后利用map函数计算[x1*y1, x2*y2, x3*y3]
        return list(map(lambda x_y: x_y[0] * x_y[1], zip(x, y)))

    @staticmethod
    def element_add(x, y):
        """
        将两个向量x和y按元素相加
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        # 变成[(x1,y1),(x2,y2),(x3,y3),...]
        # 然后利用map函数计算[x1+y1, x2+y2, x3+y3]
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def scala_multiply(v, s):
        """
        将向量v中的每个元素和标量s相乘
        """
        return map(lambda e: e * s, v)


class Perceptron(object):
    def __init__(self, input_num, activator):
        """
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        """
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0] * input_num
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        """
        打印学习到的权重、偏置项
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        输入向量，输出感知器的计算结果
        """
        # 计算向量input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]的内积
        # 然后加上bias
        return self.activator(
            VectorOp.dot(input_vec, self.weights) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        """
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        """
        一次迭代，把所有的训练数据过一遍
        """
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        """
        按照感知器规则更新权重
        """
        # 首先计算本次更新的delta
        # 然后把input_vec[x1,x2,x3,...]向量中的每个值乘上delta，得到每个权重更新
        # 最后再把权重更新按元素加到原先的weights[w1,w2,w3,...]上
        delta = label - output
        self.weights = VectorOp.element_add(
            self.weights, VectorOp.scala_multiply(input_vec, rate * delta))
        # 更新bias
        self.bias += rate * delta


def f(x):
    """
    定义激活函数f
    """
    return 1 if x > 0 else 0


def get_training_dataset():
    """
    基于and真值表构建训练数据
    """
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    """
    使用and真值表训练感知器
    """
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))
