# how to define network object?
# consisted of layers
# each layer should contain: nodes, theta matrix to next layer, delta theta matrix to next layer,
# z vector from previous layer, a vector from previous layer and intercept
# how to initialized a network?
# how to do fp?
# how to do bp?

# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-27
function: Gaussian Mixtures Model solved with Expectation-Maximization algorithm

Initializing: 每个参数初始化为一个很小的, 接近零的随机值. 例如采用均值为0, 总体方差项为0.01的高斯分布生成
Gradient: 成本函数中的正则化项不包含每一层的截距, 因此截距项与非截距项的梯度表达式不同
Stopping Criteria: 成本函数减小步长

ref:
1. http://deeplearning.stanford.edu/wiki/index.php/反向传导算法

"""

import numpy as np


class ANN:

    def __init__(self, x, y, target="classification", hl=[4, 5]):
        self.x = x
        self.y = y
        # convert multi-response label to vector
        if len(np.unique(self.y)) > 2:
            self.uni = np.unique(self.y)
            self.y = np.array([np.where(self.uni == i, 1, 0) for i in self.y])

        # shape and problem
        self.m, self.n = self.x.shape
        self.target = target

        # number of activation units for each layer (excl bias)
        # self.layers = hl
        # self.layers.insert(0, self.n)
        # ol = 1 if len(np.unique(self.y)) <= 2 else int(len(np.unique(self.y)))
        # self.layers.append(ol)
        layers = hl.copy()
        layers.insert(0, self.n)
        ol = 1 if len(np.unique(self.y)) <= 2 else int(len(np.unique(self.y)))
        layers.append(ol)
        self.layers = layers

        # record the fp result
        self.current_y = self.y * 0.0

        # initialize transforming matrices and biases
        mu = 0.0
        sigma = 0.01
        self.W = [np.random.normal(mu, sigma, (self.layers[i + 1], self.layers[i]))
                  for i in range(len(self.layers) - 1)]
        self.b = [np.random.normal(mu, sigma, (self.layers[i + 1])) for i in range(len(self.layers) - 1)]

        # initialize gradients
        self.delta_W = [np.zeros((self.layers[i + 1], self.layers[i])) for i in range(len(self.layers) - 1)]
        self.delta_b = [np.zeros((self.layers[i + 1])) for i in range(len(self.layers) - 1)]

        # initialize values of activation units
        self.a = [np.zeros(self.layers[i]) for i in range(len(self.layers))]

        # initialize values of deltas
        self.delta = [np.zeros(self.layers[i]) for i in range(len(self.layers))]

    def fp(self, xj):
        """
        forward-propagating for the jth sample
        :param xj: single sample
        :return: update self.a
        """
        # fp to update the activation values for each layer
        self.a[0] = xj

        for i in range(1, len(self.a)):
            z = self.W[i-1].dot(self.a[i-1]) + self.b[i-1]
            self.a[i] = 1 / (1 + np.exp(-z))
            # print(self.a[i])
        # self.current_y[j] = self.a[-1]

    def bp(self, yj):
        """
        back-propagating to update all deltas, and accumulate gradients
        :param yj: single response
        :return: update the deltas for each layer, and accumulate gradients for W & b
        """
        # output layer difference
        self.delta[-1] = np.multiply(self.a[-1] - yj,
                                     np.multiply(self.a[-1], 1 - self.a[-1]))

        # update deltas
        for i in list(np.arange(1, len(self.a) - 1)[::-1]):
            self.delta[i] = np.multiply(np.transpose(self.W[i]).dot(self.delta[i + 1]),
                                        np.multiply(self.a[i], (1 - self.a[i])))

        # accumulate gradients. gradients needs to be reset for each descent
        for i in range(len(self.a) - 1):
            self.delta_W[i] += np.array([self.delta[i + 1]]).T.dot(np.array([self.a[i]]))
            self.delta_b[i] += self.delta[i + 1]

    def get_cost(self, lamb=0.01):
        """
        :param lamb: regularization
        :return: value of cost function under current W & b
        """
        jwb = 0.5 * np.sum((self.current_y - self.y) ** 2) / self.m + \
            0.5 * lamb * sum([np.sum(i ** 2) for i in self.W])
        return jwb

    def train(self, lamb=0.01, alpha=0.05, maxiter=100):
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
                self.W[k] -= alpha * (self.delta_W[k] / self.m + lamb * self.W[k])
                self.b[k] -= alpha * (self.delta_b[k] / self.m)
            # reset gradients
            self.delta_W = [np.zeros((self.layers[i + 1], self.layers[i])) for i in range(len(self.layers) - 1)]
            self.delta_b = [np.zeros((self.layers[i + 1])) for i in range(len(self.layers) - 1)]
            # shrink step
            alpha *= 0.95

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

N = ANN(train_x, train_y)
N.train(alpha=1, maxiter=300, lamb=0.01)
N.predict(train_x)
N.test(train_x, train_y)

N.fp(train_x[0])
N.bp(train_y[0])
