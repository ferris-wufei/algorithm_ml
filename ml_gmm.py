# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-27
function: Gaussian Mixtures Model solved with Expectation-Maximization algorithm

Initializing: assign samples into k populations (different from k-means), and initialize
the gaussian distribution parameters with the estimates of the k populations
Stopping criteria: average of Euclidean distance between new and old mu vector below some threshold

ref:
1. http://cs229.stanford.edu/materials.html (Lecture Notes 7b, 8)

"""

import numpy as np


def get_dens(mu, sigma, x):
    """
    return the density value given a point from a multivariate gaussian population
    :param mu: population mean vector
    :param sigma: population var-cov matrix
    :param x: 1-d array
    :return: probability density
    """
    n = len(mu)
    # incompatible dimensions
    if sigma.shape != (n, n) or len(x) != n:
        print("incompatible dimensions of arguments")
        return None
    # not symmetric
    if sigma is sigma.T:
        print("sigma is not symmetric")
        return None
    # calc density
    p1 = 1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(sigma) ** 0.5)
    p2 = np.exp(- 0.5 * np.dot(np.dot(np.array([(x - mu)]), np.linalg.inv(sigma)), np.array([(x - mu)]).T))
    return p1 * p2


def get_estimate(x, w):
    """
    return the mean and var-cov matrix estimates for sample x with weight w
    :param x: 2-d array sample
    :param w: 1-d array weight
    :return: mean, var-cov matrix
    """
    m, n = x.shape
    # incompatible dimensions
    if len(w) != m:
        print("incompatible dimensions of arguments")
        return None

    # estimate mean
    weighed = np.multiply(x, np.repeat(np.array([w]).T, n, axis=1))
    mu = np.sum(weighed, axis=0) / np.sum(w)
    # estimate sigma
    accu = 0.0
    for i in range(m):
        accu += w[i] * np.array([(x[i, :] - mu)]).T * np.array([(x[i, :] - mu)])
    sigma = accu / np.sum(w)
    return mu, sigma


class GMM:
    def __init__(self, x, k=3, std=False):
        """
        :param x: numpy 2d array
        :param k: number of clusters
        :param std: standardize before training
        :return:
        """
        self.x = x
        self.m, self.n = x.shape  # num of samples and features

        # whether or not standardize before training
        self.mean = np.zeros(self.n)
        self.dev = np.zeros(self.n)
        self.std = std
        if self.std is True:  # to standardize
            self.mean = np.mean(self.x, axis=0)
            self.dev = self.x.std(axis=0)
            self.dev = np.where(self.dev == 0, 1, self.dev)  # exception: constant feature
            self.x = (self.x - self.mean) / self.dev

        # initialize labels
        self.k = k
        self.y = np.random.choice(np.arange(self.k), size=self.m, replace=True)
        
        # initialize gaussian parameters and weights
        pars = [get_estimate(self.x[self.y == i, :], np.repeat(1, sum(self.y == i))) for i in range(self.k)]
        self.mu = [a for a, b in pars]
        self.sigma = [b for a, b in pars]
        self.theta = [sum(self.y == i) / self.m for i in range(self.k)]
        self.w = np.random.rand(self.m, self.k)  # weight for each cluster in each column

    def train(self, th=0.001, maxiter=20):
        """
        :param th: threshold for average mu movement
        :param maxiter: limit of iterations
        :return:
        """
        for s in range(maxiter):
            mu_tracker = self.mu.copy()
            # E step, update the weight matrix self.w
            for i in range(self.m):
                for j in range(self.k):
                    self.w[i, j] = get_dens(self.mu[j], self.sigma[j], self.x[i, :]) * self.theta[j] / \
                                   np.sum([get_dens(self.mu[l], self.sigma[l], self.x[i, :]) * self.theta[l]
                                           for l in range(self.k)])
            # M step, update the theta and gaussian parameters
            self.theta = [np.sum(self.w[:, j]) / self.m for j in range(self.k)]
            for j in range(self.k):
                self.mu[j], self.sigma[j] = get_estimate(self.x, self.w[:, j])

            # evaluate the mu movement
            # print(self.mu)
            diff = np.array(self.mu) - np.array(mu_tracker)
            dist = np.sqrt(np.sum(diff ** 2, axis=1) / self.n)
            # print(dist)
            avg_dist = np.mean(dist)
            print("mu movement at step {0}: {1}".format(str(s), str(avg_dist)))
            if avg_dist < th:
                print("training complete")
                self.y = np.argmax(self.w, axis=1)
                return None

    def predict(self, x_new):
        """
        return the index of population which gives each new sample the maximum density
        :param x_new: 1d or 2d array
        :return: predicted population index
        """
        if len(x_new.shape) == 1:
            x_new = np.array([x_new])  # 1d array to 2d array (nested)
        if self.std is True:
            x_new = (x_new - self.mean) / self.dev
        y_new = []
        for r in x_new:
            density = [get_dens(self.mu[i], self.sigma[i], r) for i in range(self.k)]
            y_new.append(np.argmax(np.array(density)))
        return np.array(y_new)

# test
# train_x = np.random.randn(100, 5)
# G = GMM(train_x, std=False)
# print(G.y)
# G.train(th=0.01)
# print(G.mu)
# print(G.sigma)
# print(G.y)
#
# test_x = np.random.rand(20, 5)
# test_y = G.predict(test_x)
# print(test_y)
