# -*- coding: utf-8 -*-
"""Module docstring.

OOP 主流协同过滤算法:
1. ItemBasedExplict 基于显式得分的传统协同过滤算法
2. SVDExplicit 基于显式得分的SVD矩阵拟合算法: 使用梯度下降法
3. SVDImplicit 基于隐式得分的SVD矩阵拟合算法: 使用交叉最小二乘法 ALS

参考说明: 其中 1, 2 参考 http://blog.csdn.net/dark_scope/article/details/17228643, 经过修正和重写, 添加相似度计算选项
3 的数学原理部分参考论文 Collaborative Filtering for Implicit Feedback Datasets, 代码独立编写.

算法3与Spark MLlib中的推荐系统的默认算法类似; 区别是置信度的计算, 这里采用了log处理.

"""

from __future__ import division
import numpy as np
from numpy.random import random
from numpy.random import permutation
import pandas as pd
from numpy.linalg import inv


class ItemBasedExplict:
    """Return Explict Collaborative Filtering object

    基于显式得分的传统协同过滤算法, 包含相关系数和余弦两种相似度计算方法

    """
    def __init__(self, x):

        self.x = np.array(x)
        print("the input data size is ", self.x.shape)
        self.item_user = {}
        self.user_item = {}
        self.ave = np.mean(self.x[:, 2])  # 整体评分均值
        for i in range(self.x.shape[0]):  # 初始化评分字典
            uid = self.x[i][0]
            mid = self.x[i][1]
            rat = self.x[i][2]
            self.item_user.setdefault(mid, {})  # 添加外层key
            self.user_item.setdefault(uid, {})
            self.item_user[mid][uid] = rat  # 添加内层key-value
            self.user_item[uid][mid] = rat
        self.similarity = {}
        pass

    def sim_cal(self, m1, m2, method='corr'):  # 计算物品m1与物品m2的相似度
        self.similarity.setdefault(m1, {})
        self.similarity.setdefault(m2, {})
        self.item_user.setdefault(m1, {})  # 允许item_user以外的物品计算相似度
        self.item_user.setdefault(m2, {})
        self.similarity[m1].setdefault(m2, -1)
        self.similarity[m2].setdefault(m1, -1)

        if self.similarity[m1][m2] != -1:  # 已有评分, 直接返回
            return self.similarity[m1][m2]

        si = {}  # 获取共同评分用户
        for user in self.item_user[m1]:
            if user in self.item_user[m2]:
                si[user] = 1
        n = len(si)

        if method == 'corr':  # 相关系数相似度
            if n == 0:  # 无共同评分用户, 不纳入加权平均
                self.similarity[m1][m2] = 0
                self.similarity[m2][m1] = 0
                return 0
            if n <= 2:  # 2个以内共同评分用户, 相关系数恒等于1, 不能合理度量, 不纳入加权平均
                self.similarity[m1][m2] = 0
                self.similarity[m2][m1] = 0
                return 0
            s1 = np.array([self.item_user[m1][u] for u in si])
            s2 = np.array([self.item_user[m2][u] for u in si])
            sum1 = np.sum(s1)
            sum2 = np.sum(s2)
            sum1sqr = np.sum(s1 ** 2)
            sum2sqr = np.sum(s2 ** 2)
            psum = np.sum(s1 * s2)
            num = psum - (sum1 * sum2 / n)
            den = np.sqrt((sum1sqr - sum1 ** 2 / n) * (sum2sqr - sum2 ** 2 / n))
            if den == 0:  # 共同评分用户对其中一个物品的所有评分相同时，分母为0: 相关系数的局限性
                self.similarity[m1][m2] = 0
                self.similarity[m2][m1] = 0
                return 0
            self.similarity[m1][m2] = num / den
            self.similarity[m2][m1] = num / den
            return num / den
        elif method == 'cos':  # 余弦相似度
            if n < 2:  # 共同评分用户数<2, 无法计算余弦, 不纳入加权平均
                self.similarity[m1][m2] = 0
                self.similarity[m2][m1] = 0
                return 0
            s1 = np.array([self.item_user[m1][u] for u in si])
            s2 = np.array([self.item_user[m2][u] for u in si])
            l1 = np.sqrt(s1.dot(s1))
            l2 = np.sqrt(s2.dot(s2))
            cos_sim = s1.dot(s2) / l1 * l2
            self.similarity[m1][m2] = cos_sim
            self.similarity[m2][m1] = cos_sim
            return cos_sim

    def predict(self, uid, mid, method='corr'):  # 预测用户uid对物品mid的评分
        # 根据sim_cal中的处理, mid可以不在item_user内, 但uid必须在user_item中
        sim_accumulate = 0.0
        rat_acc = 0.0
        for item in self.user_item[uid]:
            sim = self.sim_cal(item, mid, method=method)
            if sim < 0:
                continue  # 得分为负数的不纳入加权平均
            rat_acc += sim * self.user_item[uid][item]
            sim_accumulate += sim
        if sim_accumulate == 0:  # 没有共同评分用户, 返回总体评分均值
            return self.ave
        return rat_acc / sim_accumulate

    def test(self, test_x, method='corr'):  # 对测试数据集test_x输出预测结果, 并计算均方误差
        test_x = np.array(test_x)
        output = []
        sums = 0
        print("the test data size is ", test_x.shape)
        for i in range(test_x.shape[0]):
            pre = self.predict(test_x[i][0], test_x[i][1], method=method)
            output.append(pre)
            sums += (pre - test_x[i][2]) ** 2  # 误差平方和累加
        rmse = np.sqrt(sums / test_x.shape[0])
        print("the rmse on test data is ", rmse)
        return output

    def recommend(self, uid, num=10, method='corr'):  # 推荐Top N
        result = {}
        for item in self.item_user:
            if item in self.user_item[uid]:  # 已评价的物品不再推荐
                continue
            score = self.predict(uid, item, method=method)  # 计算评分
            result[item] = score
        rankings = [(score, item) for item, score in result.items]  # 字典转为list of tuples
        rankings.sort()  # 排序
        rankings.reverse()  # 倒序
        return rankings[:num]  # Top N


class SVDExplicit:
    """Return Explict SVD object

    基于显式得分的矩阵分解算法, 使用梯度下降法优化成本函数

    """
    def __init__(self, x, k=20):  # k=分解后的特征向量长度
        self.x = np.array(x)
        self.k = k
        self.ave = np.mean(self.x[:, 2])  # 整体评分均值
        print("the input data size is ", self.x.shape)
        self.qi = {}
        self.pu = {}
        self.item_user = {}
        self.user_item = {}
        for i in range(self.x.shape[0]):  # 初始化评分字典
            uid = self.x[i][0]
            mid = self.x[i][1]
            rat = self.x[i][2]
            self.item_user.setdefault(mid, {})  # 添加外层key
            self.user_item.setdefault(uid, {})
            self.item_user[mid][uid] = rat  # 添加内层key-value
            self.user_item[uid][mid] = rat
            self.qi.setdefault(mid, random((self.k, 1)) / 10 * (np.sqrt(self.k)))  # 初始化特征向量
            self.pu.setdefault(uid, random((self.k, 1)) / 10 * (np.sqrt(self.k)))

    def predict(self, uid, mid):  # 预测用户uid对物品mid的评分
        if self.qi[mid] is None:  # 用户或物品不在x中, 取总体评分均值
            self.qi[mid] = np.zeros((self.k, 1))
            return self.ave
        if self.pu[uid] is None:
            self.pu[uid] = np.zeros((self.k, 1))
            return self.ave
        ans = np.sum(self.qi[mid] * self.pu[uid])  # 用户与物品特征向量的内积
        if ans > 10:  # 内积运算不能保证结果在评分的数值范围, 这里采用截断处理
            return 10
        elif ans < 1:
            return 1
        return ans

    def train(self, steps=20, gamma=0.04, Lambda=0.15):  # 梯度下降训练
        for step in range(steps):  # 梯度循环次数
            print('the ', step, '-th  step is running')
            rmse_sum = 0.0
            pm = permutation(self.x.shape[0])  # 顺序随机化
            for j in range(self.x.shape[0]):
                i = pm[j]
                uid = self.x[i][0]
                mid = self.x[i][1]
                rat = self.x[i][2]
                eui = rat - self.predict(uid, mid)
                rmse_sum += eui ** 2
                temp = self.qi[mid]  # 保留中间值
                self.qi[mid] += gamma * (eui * self.pu[uid] - Lambda * self.qi[mid])
                self.pu[uid] += gamma * (eui * temp - Lambda * self.pu[uid])
            gamma *= 0.93  # 收缩下降步长
            print("the rmse of this step on train data is ",
                  np.sqrt(rmse_sum / self.x.shape[0]))  # 输出本次循环的起始均方误差

    def test(self, test_x):  # 对测试数据集test_x输出预测结果, 并计算均方误差
        output = []
        sums = 0
        test_x = np.array(test_x)
        print("the test data size is ", test_x.shape)
        for i in range(test_x.shape[0]):
            pre = self.predict(test_x[i][0], test_x[i][1])
            output.append(pre)
            sums += (pre - test_x[i][2]) ** 2  # 误差平方和累加
        rmse = np.sqrt(sums / test_x.shape[0])
        print("the rmse on test data is ", rmse)
        return output

    def recommend(self, uid, num=10):  # 推荐Top N
        result = {}
        for item in self.item_user:
            if item in self.user_item[uid]:  # 已评价的物品不再推荐
                continue
            score = self.predict(uid, item)  # 计算评分
            result[item] = score
        rankings = [(score, item) for item, score in result.items]  # 字典转为list of tuples
        rankings.sort()  # 排序
        rankings.reverse()  # 倒序
        return rankings[:num]  # Top N


class SVDImplicit:
    """Return a Implicit SVD object

    基于隐式得分的SVD算法, 使用ALS交叉最小二乘法优化成本函数
    成本函数和解析表达式参见论文: Collaborative Filtering for Implicit Feedback Datasets

    """
    def __init__(self, x, k=20, alpha=0.1, epsilon=0.5):  # k为分解后的向量长度
        self.x = np.array(x)
        self.k = k
        print("the input data size is ", self.x.shape)
        self.num_users = np.unique(x[:, 0]).shape[0]
        self.num_items = np.unique(x[:, 1]).shape[0]
        self.rawdf = pd.DataFrame(np.zeros((self.num_users, self.num_items)),  # 使用pandas的行列标签功能, 便于predict
                                  index=np.unique(x[:, 0]),
                                  columns=np.unique(x[:, 1]))
        for i in range(self.x.shape[0]):
            user = self.x[i][0]
            item = self.x[i][1]
            rate = self.x[i][2]
            self.rawdf.loc[user, item] = rate
        self.rawmx = self.rawdf.values  # 暂不考虑标签, 使用矩阵运算

        self.pmx = np.where(self.rawmx > 0, 1, 0)  # 初始化隐式得分矩阵
        self.cmx = 1 + alpha * np.log(1 + self.rawmx / epsilon)  # 初始化得分置信度矩阵

        self.user_mx = np.random.randn(self.num_users, self.k)  # 初始化用户特征矩阵
        self.item_mx = np.random.randn(self.num_items, self.k)  # 初始化物品特征矩阵
        self.temp_user_mx = np.zeros((self.num_users, self.k))  # 中间变量
        self.temp_item_mx = np.zeros((self.num_items, self.k))  # 中间变量

        self.fitmx = np.zeros((self.num_users, self.num_items))  # 结果变量: 拟合后的矩阵

    def train(self, steps=10, Lambda=0.05):
        for s in range(steps):  # 交叉最小二乘的循环次数, 次数越多拟合效果越好, 然而同时0值的填充项越难以区分
            for u in range(self.pmx.shape[0]):
                cu = np.diag(self.cmx[u, :])
                pu = self.pmx[u, :].T
                self.temp_user_mx[u, :] = inv(self.item_mx.T.dot(cu).dot(self.item_mx) + Lambda * np.eye(self.k)) \
                    .dot(self.item_mx.T).dot(cu).dot(pu)  # 对每个用户向量应用解析表达式
            self.user_mx = self.temp_user_mx  # 更新用户特征矩阵
            for i in range(self.pmx.shape[1]):
                ci = np.diag(self.cmx[:, i])
                pi = self.pmx[:, i].T
                self.temp_item_mx[i, :] = inv(self.user_mx.T.dot(ci).dot(self.user_mx) + Lambda * np.eye(self.k)) \
                    .dot(self.user_mx.T).dot(ci).dot(pi)  # 对每个物品向量应用解析表达式
            self.item_mx = self.temp_item_mx  # 更新物品特征矩阵

            weighed_ss = (self.cmx * (self.pmx - self.user_mx.dot(self.item_mx.T)) ** 2).sum()  # 加权残差平方和
            regular_ss = Lambda * (np.array([i.dot(i) for i in self.user_mx]).sum() +
                                   np.array([j.dot(j) for j in self.item_mx]).sum()) ** 2  # 正则化项
            cost = weighed_ss + regular_ss  # 成本函数的值
            print("value of cost function:", cost)

        self.fitmx = self.user_mx.dot(self.item_mx.T)  # 拟合后的矩阵
        self.fitpd = pd.DataFrame(self.fitmx, index=self.rawdf.index, columns=self.rawdf.columns)  # 拟合后的数据框

    def predict(self, user, item):
        if (user in self.fitpd.index) and (item in self.fitpd.columns):
            return self.fitpd.loc[user, item]
        else:
            print("user and item must be in the training data")  # 用户和物品必须存在初始化矩阵中
            return None
    # recommend函数可参照SVDExplicit或SVDImplicit
