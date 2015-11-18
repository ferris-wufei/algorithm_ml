# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-13
function: 利用cvxpy求解二次规划, 对linear SVM和kernel SVM的Primal Problem求解
ref:
Training a Support Vector Machine in the Primal
http://www.cs.utah.edu/~piyush/teaching/svm-solving-primal.pdf
Implementing linear SVM using quadratic programming (Toby Dylan Hocking)
to-do:
1. 记录支持向量
2. 决策边界二维图


"""
import numpy as np
import cvxpy as cvx

# 支持向量机OOP


class SVM:
    def __init__(self, x, y, kernel="linear"):
        """

        :param x: 训练数据
        :param y: 训练标签
        :param kernel: 核选项
        :return: None
        """
        if kernel not in ("linear", "gaussian", "polynomial"):
            print("invalid kernel option, please use one of these:"
                  "linear, gaussian, polynomial")
            return None
        self.kernel = kernel  # 核选项
        self.k_func = None  # 核函数
        self.x = x
        self.y = y
        self.m = x.shape[0]  # 训练样本数量
        self.n = x.shape[1]  # 特征数量
        self.k = np.zeros((self.m, self.m))  # 核矩阵
        if len(y.shape) == 1:  # 如果y是1维向量, 则转化为mx1矩阵
            self.y = np.array([[i] for i in y])
        else:
            self.y = y
        self.w = []  # linear SVM的系数向量
        self.a = []  # kernel SVM的系数向量
        self.b = 0  # linear & kernel SVM公用的截距项

    def train(self, ct=0.01, theta=1, p=2):
        """

        :param ct: 松弛变量正则化系数
        :param theta: 高斯核函数参数
        :param p: 多项式核函数参数
        :return: None
        """
        def kg(a, b, value_theta=theta):  # 高斯核函数
            sim = np.exp( -1 * (a - b).dot(a - b) / (2 * value_theta ** 2))
            # sim = np.dot(a, b)
            return sim

        def kp(a, b, value_p=p):  # 多项式核函数
            sim = (np.dot(a, b) + 1) ** value_p
            return sim

        def k_mat(x, k_func=kg):  # 核矩阵生成函数
            m = x.shape[0]
            mat = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    mat[i, j] = k_func(x[i, :], x[j, :])
            return mat

        if self.kernel == "linear":  # 线性核
            w = cvx.Variable(self.n)
            e = cvx.Variable(self.m)
            b = cvx.Variable()
            c = cvx.Parameter(sign="positive")
            c.value = ct

            obj = cvx.Minimize(0.5 * cvx.norm(w,2) + c * cvx.sum_entries(e))
            constraints = [e >= 0,
                           cvx.mul_elemwise(self.y, self.x * w + b) - 1 + e >= 0]
            prob = cvx.Problem(obj, constraints)

            prob.solve()
            self.w = np.array(w.value)
            self.b = b.value
            return None

        elif self.kernel == "gaussian":  # 高斯核
            self.k = k_mat(self.x, k_func=kg)
            self.k_func = kg
        else:  # 多项式核
            self.k = k_mat(self.x, k_func=kp)
            self.k_func = kp

        a = cvx.Variable(self.m)
        b = cvx.Variable()
        e = cvx.Variable(self.m)
        c = cvx.Parameter(sign="positive")
        c.value = ct

        obj = cvx.Minimize(0.5 * cvx.quad_form(a, self.k) + c * cvx.sum_entries(e))
        constraints = [e >= 0,
                       self.y * b + cvx.mul_elemwise(self.y, self.k * a) - 1 + e >= 0]
        prob = cvx.Problem(obj, constraints)  # quadratic programming 二次规划
        prob.solve()
        self.a = np.array(a.value)
        self.b = b.value
        return None

    def predict(self, x_new):
        """

        :param x_new: 待预测数据
        :return: 预测正例负例标签
        """
        if self.kernel == "linear":
            nested = np.where(x_new.dot(self.w) + self.b >= 0, 1, -1)
            flatten = [i[0] for i in nested]  # 列矩阵转化为向量
            return np.array(flatten)
        else:
            pred = []
            for r in range(x_new.shape[0]):
                x1 = x_new[r, :]  # 每次对一个样本预测
                y1 = 0
                for i in range(self.m):
                    y1 += self.k_func(x1, self.x[i, :]) * self.a[i]
                y1 += self.b
                pred.append(y1)
            nested = np.where(np.array(pred) >= 0, 1, -1)
            flatten = [i[0] for i in nested]
            return np.array(flatten)

    def test(self, x_test, y_test):
        """

        :param x_test: 测试数据
        :param y_test: 测试标签
        :return: 错误率
        """
        y_pred = self.predict(x_test)
        error_ratio = 1.0 * np.sum(y_test != y_pred) / len(y_test)
        print("test error rate: {0}".format(error_ratio))  # 暂时只输出错误率
        return error_ratio


# # generate test data a and label b
# x1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50)
# x2 = np.random.multivariate_normal([1, 3], [[1, 0], [0, 1]], 50)
# y1 = np.repeat(1, 50)
# y2 = np.repeat(-1, 50)
# a = np.row_stack((x1, x2))
# b = np.concatenate((y1, y2))
