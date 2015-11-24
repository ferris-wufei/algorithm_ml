# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-21
function: Adaboost判别模型, 基于ml_decision_trees.py中CART算法

"""

import numpy as np
import ml_decision_tree as dts


class AdaBoost:
    def __init__(self, rows):
        self.rows = rows
        self.y = [r[len(rows[0]) - 1] for r in rows]  # 样本y标签
        self.trees = []
        self.m = len(rows)  # 样本容量
        self.alpha = None

    def train(self, k=50, th=0.0, d=2):
        """
        循环训练, 得到self.trees和self.alpha
        :param k: 循环次数
        :param th: 阈值传递给dts.train_cart
        :param d: 树的深度, 传递给dts.train_cart
        :return:
        """
        self.alpha = np.repeat([0], k)  # 初始化树权重
        weight = np.repeat([1 / self.m], self.m)  # 初始化样本权重

        for i in range(k):

            # 首次训练
            # bootstrap抽样
            sample_indices = np.random.choice(range(self.m), size=self.m, replace=True, p=weight)
            sampled = [self.rows[i] for i in sample_indices]
            tree = dts.train_cart(sampled, th=th, d=d, sample=False)
            predicted = dts.predict(tree, self.rows, out="value")  # 预测原样本
            err_vec = np.where(np.array(self.y) != np.array(predicted), 1, 0)
            err_rate = (1 / self.m) * err_vec.dot(weight)  # 错误率

            # 错误率超过0.5, 重新训练
            while err_rate > 0.5:
                weight = np.repeat([1 / self.m], self.m)  # 重置权重
                # bootstrap抽样
                sample_indices = np.random.choice(range(self.m), size=self.m, replace=True, p=weight)
                sampled = [self.rows[i] for i in sample_indices]
                tree = dts.train_cart(sampled, th=th, d=d, sample=False)
                predicted = dts.predict(tree, self.rows, out="value")  # 预测原样本
                err_vec = np.where(np.array(self.y) != np.array(predicted), 1, 0)
                err_rate = (1 / self.m) * err_vec.dot(weight)  # 错误率

            self.alpha[i] = 0.5 * np.log((1 - err_rate) / err_rate)  # 更新树权重
            weight *= np.exp(self.alpha[i] * err_vec)  # 更新样本权重
            weight /= np.sum(weight)  # 归一化样本权重
            self.trees.append(tree)  # 树入栈

    def predict(self, rows_new):
        """
        对self.trees的每个预测结果, 按照self.alpha累加权重, 返回权重最高的y标签
        :param rows_new: 待预测数据
        :return: adaboost预测y标签
        """
        results = []
        for r in rows_new:
            y_dic = {}  # key: 预测y标签; value: 树权重
            for i in range(len(self.trees)):
                t = self.trees[i]
                p = dts.predict_single(t, r, out="value")
                y_dic[p] = y_dic.get(p, 0) + self.alpha[i]  # 累加每个y标签值的权重
            weighed_p = dts.topkey(y_dic)
            results.append(weighed_p)
        return results

    def test(self, rows_test):
        """
        :param rows_test: 待测试数据
        :return: 错误率
        """
        l = len(rows_test[0]) - 1
        predicted = self.predict(rows_test)
        actual = [r[l] for r in rows_test]
        err = sum(np.array(predicted) != np.array(actual))
        err_rate = float(err) / len(rows_test)
        print("error rate: {0}".format(str(err_rate)))
        return err_rate


# ad = AdaBoost(dts.my_data)
# ad.train(k=30)
# ad.predict(dts.my_data)
# ad.test(dts.my_data)
