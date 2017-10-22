# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd

from python3.perceptron import Perceptron


# 定义激活函数f
def active_function(x):
    # return x
    if x < 0.5:
        return 0
    elif x > 0.5 and x < 2:
        return 1
    elif x > 2:
        return 2


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数'''
        Perceptron.__init__(self, input_num, activator_fun=active_function)


def get_training_dataset():
    '''
    捏造5个人的收入数据
    '''
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    data = pd.read_csv('iris.data', header=None)
    # take [0, 1] columns of data to be x, take 4 column and do categorical codes as y.
    # y now is an array composed by 0 (stands for Iris-setosa), 1 (stands for Iris-versicolor), and
    # 2 (stands for Iris-virginica)
    x, y = data[[0, 1, 2, 3]], pd.Categorical(data[4]).codes
    # pd.to_list

    # print(x.values.tolist())
    input_vecs = [x.values.tolist()][0]
    labels = y.tolist()
    print(type(labels), labels)
    # labels = y.
    print(input_vecs)
    # 期望的输出列表，月薪，注意要与输入一一对应
    # labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(4)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10000, 0.001)
    # 返回训练好的线性单元
    return lu


# def plot(linear_unit):
#     import matplotlib.pyplot as plt
#     input_vecs, labels = get_training_dataset()
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(map(lambda x: x[0], input_vecs), labels)
#     weights = linear_unit.weights
#     bias = linear_unit.bias
#     x = range(0,12,1)
#     y = map(lambda x:weights[0] * x + bias, x)
#     ax.plot(x, y)
#     plt.show()


if __name__ == '__main__':
    '''训练线性单元'''

    # data = pd.read_csv('iris.data', header=None)
    # take [0, 1] columns of data to be x, take 4 column and do categorical codes as y.
    # y now is an array composed by 0 (stands for Iris-setosa), 1 (stands for Iris-versicolor), and
    # 2 (stands for Iris-virginica)
    # x, y = data[[0, 1, 2, 3]], pd.Categorical(data[4]).codes
    # pd.to_list

    # print(x, y)
    linear_unit = train_linear_unit()
    print("predict")
    print(linear_unit.predict([5.1, 3.5, 1.4, 0.2]))
    print(linear_unit.predict([5.9, 3.0, 5.1, 1.8]))
    print(linear_unit.predict([6.0,2.2,5.0,1.5]))
    print(linear_unit.predict([6.0, 3.4, 4.5, 1.6]))

    # 打印训练获得的权重
