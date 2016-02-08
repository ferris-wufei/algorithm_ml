# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-22
function: Gradient Boosting Decision Tree
A regression model based on ml_decision_tree. Here I present the most common situation of
gradient boosting, in which the loos function is sum of squared errors.
对于广义的GB, 梯度(伪残差)和回归树的分裂条件都要修改.
ref:
1. 李航: 统计学习方法 P151
2. The Elements of Statistical Learning P380

"""

import numpy as np
import ml_DTS as dts


class GBDT:
    def __init__(self, rows):
        self.rows = rows
        self.y = np.array([r[len(rows[0]) - 1] for r in rows], dtype='float64')  # 样本y值
        self.trees = []
        self.m = len(rows)  # 样本容量
        self.resid = self.y  # 初始化残差
        self.rows_updated = self.rows  # 初始化提升训练数据

    def train(self, d=1, k=20):
        """
        回归问题的提升树算法
        :param d: 每棵树的深度
        :param k: 树的数量
        :return:
        """

        def update_y(rows, y_new):
            """
            替换数据集rows的y值, 返回一个新的数据集
            :param rows: 数据集
            :param y_new: 新的y值, list-like
            :return:
            """
            updated = []
            col_id = len(rows[0]) - 1  # 列数
            n_rows = len(rows)  # 行数
            for l in range(n_rows):
                r_new = rows[l][:col_id]
                r_new.append(y_new[l])
                updated.append(r_new)
            return updated

        for i in range(k):
            # 用残差更新训练数据的y值
            self.rows_updated = update_y(self.rows_updated, self.resid)
            # 训练新的weak learner, 入栈
            t = dts.train_cart(self.rows_updated, target="regression", d=d)
            self.trees.append(t)
            # 更新残差
            pred = dts.predict(t, self.rows_updated)
            # print(np.array(pred))
            # print(self.resid)
            self.resid -= np.array(pred)
            # 输出当前步骤均方误差
            mse = np.dot(self.resid, self.resid) / len(self.resid)
            print("training step {0} finished, current mse: {1}".format(str(i), str(mse)))

    def predict(self, rows_new):
        """
        将self.trees的每个预测结果累加, 得到最终预测值
        :param rows_new: 待预测数据, 嵌套的list或tuple
        :return: gbdt预测的y值
        """
        result = []
        for r in rows_new:
            y_new = 0.0
            # 预测结果为所有树的结果累加
            for t in self.trees:
                y_new += dts.predict_single(t, r)
            result.append(y_new)
        return result

    def test(self, rows_test):
        """
        :param rows_test: 待测试数据, 嵌套的list或tuple
        :return: 均方误差
        """
        l = len(rows_test[0]) - 1
        predicted = self.predict(rows_test)
        actual = [r[l] for r in rows_test]
        error = np.array(predicted) - np.array(actual)
        mse = np.dot(error, error) / len(error)
        print("test mse: {0}".format(str(mse)))
        return mse


# # test
# new_data = update_y(dts.my_data, [0, 2, 1, 1, 2, 0, 1, 2, 0, 0, 0, 0, 1, 0, 1, 1])
# g = GBDT(new_data)
# g.train(k=20)
# g.predict(new_data)
# g.test(new_data)
