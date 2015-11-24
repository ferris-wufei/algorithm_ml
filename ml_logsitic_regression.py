# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-15
function: 利用凸优化库cvxpy, 实现标准, L1和L2正则化 (共3种) 逻辑回归算法

"""

import cvxpy as cvx
import numpy as np


# 逻辑回归对象
class LR:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = x.shape[0]  # 训练样本数量
        self.n = x.shape[1]  # 特征数量
        self.x_trans = np.column_stack((x, np.repeat(1, self.m)))  # 添加数字1列, 对应截距参数w0
        self.y_trans = np.array([[i] for i in y])  # 向量转化为mx1矩阵
        self.w = []  # 参数向量w, 对应算法说明中的theta

    def train(self, level=0, lamb=0.01):
        """

        :param level: 0: 非正则化; 1: 1阶正则化; 2: 2阶正则化
        :param lamb: 正则化系数水平
        :return: 无
        """
        L = cvx.Parameter(sign="positive")
        L.value = lamb  # 正则化系数
        w = cvx.Variable(self.n + 1)  # 参数向量
        loss = 0
        for i in range(self.m):  # 构造成本函数和正则化项
            loss += self.y_trans[i] * \
                    cvx.log_sum_exp(cvx.vstack(0, cvx.exp(self.x_trans[i, :].T * w))) + \
                    (1 - self.y_trans[i]) * \
                    cvx.log_sum_exp(cvx.vstack(0, cvx.exp(-1 * self.x_trans[i, :].T * w)))
        # 为什么一定要用log_sum_exp? cvx.log(1 + cvx.exp(x[i, :].T * w))为什么不行?
        if level > 0:
            reg = cvx.norm(w[:self.n], level)
            prob = cvx.Problem(cvx.Minimize(loss / self.m + L / (2 * self.m) * reg))
        else:
            prob = cvx.Problem(cvx.Minimize(loss / self.m))
        prob.solve()
        self.w = np.array(w.value)
        
    def predict(self, x_new, threshold=0.5):
        """

        :param x_new: 待预测数据集
        :param threshold: logit判定阈值
        :return: 预测标签
        """
        mt = x_new.shape[0]
        x_new = np.column_stack((x_new, np.repeat(1, mt)))  # 添加数字1列, 对应截距参数w0
        y_prob = 1 / (1 + np.exp(x_new.dot(self.w)))  # logistic输出
        y_pred = np.where(y_prob >= 0.5, 1, 0)  # 根据threshold判断正例负例
        return y_pred
    
    def test(self, x_new, y_new, threshold=0.5):
        """

        :param x_new: 待检验数据集
        :param y_new: 检验标签
        :param threshold: logit检验阈值
        :return: 错误率
        """
        y_pred = self.predict(x_new, threshold=threshold)
        y_new_trans = np.array([[i] for i in y_new])
        err_rate = np.sum(y_new_trans != y_pred) / y_new_trans.shape[0]
        print("test error rate: {0}".format(err_rate))
        return err_rate

# test
# x1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50)
# x2 = np.random.multivariate_normal([1, 3], [[1, 0], [0, 1]], 50)
# y1 = np.repeat(1, 50)
# y2 = np.repeat(0, 50)
# a = np.row_stack((x1, x2))
# b = np.concatenate((y1, y2))
#
# L_test = LR(a, b)
# L_test.train(level=2, lamb=0.01)
# print(L_test.w)
# L_test.predict(a[:10, :])
# L_test.test(a[:10, :], b[:10])
