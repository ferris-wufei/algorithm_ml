# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-27

Initializing: 每个参数初始化为一个很小的, 接近零的随机值. 这里采用高斯分布
Gradient: 成本函数中的正则化项不包含每一层的截距, 因此截距项与非截距项的梯度表达式不同
Stopping Criteria: 成本函数减小步长

ref:
1. http://deeplearning.stanford.edu/wiki/index.php/反向传导算法
2. http://neuralnetworksanddeeplearning.com/chap2.html

"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NN:

    def __init__(self, x, y, target="classification", hl=[4, 5]):
        self.x = x
        self.y = y
        # convert multi-response label to vector
        self.uni = np.unique(self.y)
        if len(self.uni) > 2:
            self.y = np.array([np.where(self.uni == i, 1, 0) for i in self.y])

        # shape and problem
        self.m, self.n = self.x.shape
        self.target = target

        # number of activation units for each layer (excl bias)
        layers = hl.copy()
        layers.insert(0, self.n)
        ol = 1 if len(self.uni) <= 2 else int(len(self.uni))
        layers.append(ol)
        self.layers = layers
        self.L = len(self.layers)

        # record the fp result
        self.current_y = self.y * 0.0

        # initialize weights and biases
        self.W = [np.random.randn(i, j) / np.sqrt(i) for i, j in zip(self.layers[1:], self.layers[:-1])]
        self.b = [np.random.randn(i, 1) for i in self.layers[1:]]

        # initialize gradients
        self.delta_W = [np.zeros((i, j)) for i, j in zip(self.layers[1:], self.layers[:-1])]
        self.delta_b = [np.zeros((i, 1)) for i in self.layers[1:]]

        # initialize activations
        self.a = [np.zeros((i, 1)) for i in self.layers]
        
        # initialize zs
        self.z = [np.zeros((i, 1)) for i in self.layers[1:]]

        # initialize deltas
        self.delta = [np.zeros((i, 1)) for i in self.layers[1:]]

    def fp(self, xj):
        """
        forward-propagating for the jth sample
        :param xj: single sample
        :return: update self.a
        """
        # convert input layer(1st activation to column array)
        self.a[0] = xj[np.newaxis].T

        # update z & activations of each layer
        for i in range(self.L - 1):
            z = np.dot(self.W[i], self.a[i]) + self.b[i]
            self.z[i] = z
            self.a[i + 1] = sigmoid(z)

    def bp(self, yj):
        """
        back-propagating to update all deltas, and accumulate gradients
        :param yj: single response
        :return: update the deltas for each layer, and accumulate gradients for W & b
        """
        # gradients of output layer
        self.delta[-1] = (self.a[-1] - yj) # * sigmoid_prime(self.z[-1])
        self.delta_b[-1] += self.delta[-1]
        self.delta_W[-1] += np.dot(self.delta[-1], self.a[-2].T)

        # gradients of previous layers
        for l in range(2, self.L):
            z = self.z[-l]
            sp = sigmoid_prime(z)
            self.delta[-l] = np.dot(self.W[-l + 1].T, self.delta[-l + 1]) * sp
            self.delta_b[-l] += self.delta[-l]
            self.delta_W[-l] += np.dot(self.delta[-l], self.a[-l - 1].T)

    def get_cost(self, lamb=0.0001):
        """
        using cross_entropy as cost instead of squared errors
        attention: keep lamb small otherwise the training will never converge
        :param lamb: regularization
        :return: value of cost function under current W & b
        """
        cross_entropy = - np.sum(self.y * np.log(self.current_y) + (1 - self.y) * np.log(1 - self.current_y)) / self.m
        regular_term = 0.5 * lamb * sum([np.sum(i ** 2) for i in self.W]) / self.m
        return cross_entropy + regular_term

    def train(self, lamb=0.0001, alpha=0.05, maxiter=100):
        """
        what's the appropriate stopping criteria???
        :param lamb: regularization
        :param alpha: gradient step
        :param maxiter: number of iterations
        :return:
        """
        for i in range(maxiter):
            # fp & bp for each sample
            for j in range(self.m):
                self.fp(self.x[j])
                self.current_y[j] = self.a[-1]
                self.bp(self.y[j])
            # evaluate cost
            cost = self.get_cost(lamb=lamb)
            print("cost before update: {0}".format(str(cost)))
            # gradient descent
            for k in range(len(self.W)):
                self.W[k] -= alpha * (self.delta_W[k] / self.m + lamb * self.W[k] / self.m)
                self.b[k] -= alpha * (self.delta_b[k] / self.m)
            # reset gradients
            self.delta_W = [np.zeros((i, j)) for i, j in zip(self.layers[1:], self.layers[:-1])]
            self.delta_b = [np.zeros((i, 1)) for i in self.layers[1:]]
            # shrink step
            # alpha *= 0.99

        print("training complete")

    def predict(self, x_new):
        """
        :param x_new: 2d array
        :return:
        """
        if len(x_new.shape) == 1:
            x_new = np.array([x_new])  # 1d array to 2d array (nested)
        y_new = []
        for r in x_new:
            self.fp(r)
            y_new.append(self.a[-1])
        return np.array(y_new)

    def test(self, x_test, y_test):
        """
        :param x_test: test data
        :param y_test: test label
        :return:
        """
        y_new = self.predict(x_test)

        if len(np.unique(y_test)) > 2:
            y_test = np.array([np.where(self.uni == i, 1, 0) for i in y_test])

        counter = 0.0
        for i in range(len(y_test)):
            if y_new[i] != y_test[i]:
                counter += 1
        err_rate = counter / len(y_test)
        return err_rate

# test procedure
x1 = np.random.multivariate_normal([0, 0, 0], [[2, 0, 1], [0, 3, 0], [1, 0, 2]], 50)
x2 = np.random.multivariate_normal([1, 0.5, 2], [[1, 0.5, 0.5], [0.5, 2, 1], [0.5, 1, 1]], 50)
y1 = np.repeat(1, 50)
y2 = np.repeat(0, 50)
train_x = np.row_stack((x1, x2))
train_y = np.concatenate((y1, y2))

N = NN(train_x, train_y)
N.train(alpha=10, maxiter=500, lamb=0.0001)
N.predict(train_x)
N.test(train_x, train_y)
