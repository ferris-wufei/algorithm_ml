# -*- coding: utf-8 -*-
"""Module docstring.

author: ferris
update: 2015-11-19
function: 基于ml_decision_trees.py的随机森林算法. 主要步骤:
1. bootstrap抽样
2. 训练: 随机化候选特征, 逐个训练树, 保存到森林
3. 预测: 遍历森林, 逐个预测, 结果保存到列表

"""
import numpy as np
import ml_decision_trees as dts


class RandomForrest:
    def __init__(self, rows, target="classification"):
        self.rows = rows
        self.trees = []
        self.target = target  # 判别或回归
        self.m = len(rows)  # 样本容量

    def train(self, num_trees=200, num_features=2, threshold=0.0):
        """
        逐个训练, 采用cart算法
        :param num_trees: 树的数量
        :param num_features: 每个划分随机抽取的候选特征数量
        :param threshold: 传递给dts.train
        :return:
        """
        for i in range(num_trees):
            # bootstrap有放回抽样
            sample_indices = np.random.choice(range(self.m), size=self.m, replace=True)
            sampled = [self.rows[i] for i in sample_indices]
            # 根据ml_decision_trees.train的设定, 当m=num_features时, 即使sample=True, 也会执行bagging
            tree = dts.train(sampled, threshold=threshold,
                             target=self.target, m=num_features, sample=True)
            self.trees.append(tree)
        print("training complete on {0} trees".format(num_trees))

    def predict(self, rows_new):
        """

        :param rows_new: 待预测数据集
        :return: 判别: 投票最高标签; 回归: 预测结果均值
        """
        results = []
        for r in rows_new:
            sub_results = []
            for t in self.trees:
                predicted = dts.predict_single(t, r, out="value")
                sub_results.append(predicted)
            if self.target == "classification":  # 判别: 投票
                tmp_dict = {}
                for i in sub_results:
                    tmp_dict[i] = tmp_dict.get(i, 0) + 1
                k_tracker = None
                v_tracker = 0.0
                for k, v in tmp_dict.items():
                    if v > v_tracker:
                        k_tracker = k
                        v_tracker = v
                results.append(k_tracker)
            else:  # 回归: 求均值
                results.append(np.mean(sub_results))
        return results

    def test(self, rows_test):
        """

        :param rows_test: 验证数据集
        :return: 判别: 错误率; 回归: 均方误差
        """
        l = len(rows_test[0]) - 1
        predicted = self.predict(rows_test)
        actual = [r[l] for r in rows_test]
        if self.target == "classification":
            err = sum(np.array(predicted) != np.array(actual))
            err_rate = float(err) / len(rows_test)
            return err_rate
        else:
            rss = sum((np.array(predicted) - np.array(actual)) ** 2)
            mse = float(rss) / len(rows_test)
            return mse

# # test
# dt = dts.my_data
# rf = RandomForrest(dt)
# rf.train(num_features=3)
# rf.predict(dt)
# rf.test(dt)
