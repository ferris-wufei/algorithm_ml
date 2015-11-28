# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-26
function: k-means clustering.

Initializing: assign k centroids out of the samples randomly
Stopping criteria: average of Euclidean distance between new and old centroids below some threshold.

ref:
1. http://cs229.stanford.edu/materials.html (Lecture Notes 7a)

"""

import numpy as np


class KM:

    def __init__(self, x, k=3, std=False):
        """
        :param x: numpy 2d array
        :param k: number of clusters
        :param std: standardize before training
        :return:
        """
        self.x = x
        self.k = k
        self.m, self.n = x.shape  # num of samples and features
        self.y = np.zeros(self.m)  # labels
        self.rand_id = np.random.choice(np.arange(self.m), size=self.k)
        self.centroid = self.x[self.rand_id, :]  # initialize centroid
        self.cluster = [[] for i in range(k)]  # initialize cluster
        self.dist = np.zeros((self.m, self.k))  # m*k matrix of each sample, each centroid
        # whether or not standardize before training
        self.mu = np.zeros(self.n)
        self.dev = np.zeros(self.n)
        self.std = std
        if self.std is True:  # to standardize
            self.mu = np.mean(self.x, axis=0)
            self.dev = self.x.std(axis=0)
            self.dev = np.where(self.dev == 0, 1, self.dev)  # exception: constant feature
            self.x = (self.x - self.mu) / self.dev

    def train(self, th=0.01, maxiter=20):
        """
        :param th: threshold for average centroid movement
        :param maxiter: limit of iterations
        :return:
        """
        # centroid tracker
        centroid_tracker = self.centroid

        for i in range(maxiter):
            for j in range(self.k):  # update distance
                sample_dist = self.x - self.centroid[j]
                self.dist[:, j] = np.sqrt(np.sum(sample_dist ** 2, axis=1) / self.n)
            self.y = np.argmin(self.dist, axis=1)  # update cluster label
            self.centroid = [np.mean(self.x[self.y == l, :], axis=0) for l in range(self.k)]  # update centroids
            self.cluster = [self.x[self.y == l, :] for l in range(self.k)]  # update cluster samples
            # stopping criteria
            centroid_diff = np.array(centroid_tracker) - np.array(self.centroid)
            centroid_dist = np.sqrt(np.sum(centroid_diff ** 2, axis=1) / self.n)
            avg_dist = np.mean(centroid_dist)  # average of centroid distance movement
            print("centroid movement at step {0}: {1}".format(str(i), str(avg_dist)))
            centroid_tracker = self.centroid
            if avg_dist < th:
                print("clustering complete")
                return None

        print("clustering complete")
        return None

    def predict(self, x_new):
        """
        :param x_new: 1d or 2d array
        :return: predicted cluster index
        """
        if len(x_new.shape) == 1:
            x_new = np.array([x_new])  # 1d array to 2d array (nested)
        if self.std is True:
            x_new = (x_new - self.mu) / self.dev
        y_new = []
        for r in x_new:
            k_diff = r - self.centroid
            k_dist = np.sqrt(np.sum(k_diff ** 2, axis=1) / self.n)
            r_y = np.argmin(k_dist)
            y_new.append(r_y)
        return np.array(y_new)

# test
# train_x = np.random.randn(100, 5)
# K = KM(train_x, std=True)
# K.train(th=0.01)
# print(K.centroid)
# print(K.y)
# print([len(i) for i in K.cluster])
#
# test_x = np.random.rand(5)
# test_y = K.predict(test_x)
# print(test_y)
