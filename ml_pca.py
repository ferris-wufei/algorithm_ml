# -*- coding: utf-8 -*-
"""
author: ferris
update: 2015-12-13
function: Principle Component Analysis
"""

import numpy as np


class PCA:
    def __init__(self, x, std=True):
        """
        :param x: 2d numpy array with each row for a sample
        :param std: True for standardize before training
        :return:
        """
        self.m, self.n = x.shape
        self.x = x
        self.std = std

        # standardize and store the intermediate vars
        if std is True:
            self.mu = np.mean(x, axis=0)
            self.dev = np.std(x, axis=0)
            self.x1 = (self.x - self.mu) / self.dev

        # store the eigenvalues & eigenvectors
        self.l = None
        self.e = None

    def train(self):
        """
        calculate eigenvalues & eigenvectors of sample covariance matrix
        arrange them according to descending order of eigenvalues
        :return:
        """
        mi = np.eye(self.m)  # identity matrix
        m1 = np.matrix(np.ones(self.m)).T  # matrix of ones

        # sample covariance matrix
        sigma = 1 / (self.m - 1) * np.dot(np.dot(self.x.T, mi - (1 / self.m) * np.dot(m1, m1.T)), self.x)

        # eigenvalues & eigenvectors
        self.l, self.e = np.linalg.eig(sigma)

        # sort by eigenvalues descending
        sort = np.argsort(self.l)[::-1]  # descending index
        self.l = self.l[sort]
        self.e = self.e[sort]

    def predict(self, x_new, pct=0.8):
        """
        :param x_new: 2d numpy array with same number of features as self.x, could be self.x
        :param pct: how much of total variance explained by selected components
        :return:
        """
        # number of components explaining pct of total variance
        pos = pct * np.sum(self.l)
        num_comp = self.l.cumsum().searchsorted(pos) + 1
        # calculate the first components
        z = self.e[range(num_comp)]
        y_new = np.dot(x_new, z.T)
        return y_new


# # demo
# x0 = np.random.rand(100, 10)
# P = PCA(x0)
# P.train()
# y0 = P.predict(x0)
