# (note) 支持向量机SVM算法与Python实现

@(Math)[machine learning|math]

[TOC]


## 1. 线性核

这里仅讨论SVM的Primal形式，并直接使用QP优化求解，并不需要理解Lagrange对偶问题和SMO算法。关于优化表达式的推导，可以参考：

> Implementing linear SVM using quadratic programming (by Toby Dylan Hocking)
> Coursera 上 Andrew Ng 的 Machine Learning 教程
> http://www.cnblogs.com/jerrylead/archive/2011/03/13/1982639.html

### 1.1 成本函数

先考虑线性可分的情况。假设样本数为 $m$，线性决策边界的形式为 $y = w \cdot x + b$，我们的目标是找到一个边界，使距离边界最近的点的距离最大化，即：

$$\max_{w, b} \min_i||w \cdot x_i + b||$$

由于我们可以随意选定比例，缩放边界参数的范数 $||w||, ||b||$，因此以上问题等价于：**固定最近的点的距离为1，找到合适的 $w, b$，使得最近距离点对应的 $x_i$ 值最大化，即最小化 $||w||$。（这一段需要仔细理解）**

引入松弛变量 $\xi_i,\ i=1,...,m$，我们对全体样本的松弛变量的1阶或2阶矩进行约束，结合上面的等价问题，得到最优化目标函数：

$$\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^m \xi_i^p, \ p=1,2$$
$$s.t. \forall i, \ y_i (w \cdot x_i + b) \ge 1 - \xi_i; \ \xi_i \ge 0$$

之后的Python代码，采用的就是以上目标函数表达式。其中 $p=1$ 对应linear loss，$p=2$ 对应quadratic loss，是对满足 $y_i (w \cdot x_i + b) \lt 1$ 样本的两种不同的成本函数，如下图所示。

![Alt text](./1447768394907.png)


### 1.2 与逻辑回归的关系

在Andrew Ng的课程中，SVM可以理解为逻辑回归的一个推广情形。沿用原讲义的符号 $w \cdot x + b = \theta \cdot x$，将逻辑回归的成本函数修改为：

$$J_{\theta} = C \sum_{i=1}^m [y_i cost(\theta^T x_i)] + \frac{1}{2} \sum_{j=1}^n \theta_j$$

这里为了表达式的简洁，将逻辑回归中正例反例的1/0值改为1/-1。其中成本函数项 $cost(z)$ 在正例下的曲线同上图。

考虑线性可分的情况，当 $C$ 取一个很大的值，$\theta$ 就需要取合适的值使 $J_{\theta} $ 的第一项为0，于是优化问题转化为：

$$min_{\theta} \frac{1}{2} \sum_{j=1}^n {\theta_j}^2$$
$$ s.t. \forall i, \ y_i\theta^T x_i \ge 1$$

引入松弛变量后，优化问题与以上等价。

> SVM与逻辑回归的重要差异：
> - 先看SVM，从 $cost(z)$ 的曲线可以看出，只要约束条件得到满足，$J_\theta$ 的第一项就为0；几何上，等价于在margin之外的点，无论距离决策边界有多远，对 $J_\theta$ 都没有任何影响。即决策边界仅与附近的点有关。
> - 与之相对，在逻辑回归中，当任意一个点进一步远离边界时，都会使 $J_\theta$ 进一步缩小，即决策边界与所有的点有关。因此已经远离边界的点，可以对 $J_\theta$ 施加影响，而牺牲决策边界对距离较近的点的判断能力。


## 2. 其他核

### 2.1 成本函数

对于Kernel SVM，预测函数改写为 $y = \sum_{i=1}^m \alpha_i k(x, x_i) + b$，而含松弛变量的目标函数为：

$$\min_\alpha \frac{1}{2} \alpha^T K \alpha + C \sum_{i=1}^m \xi_i^p, \ p=1,2$$
$$s.t. \forall i, \ y_i (\alpha^T K_i + b) \ge 1 - \xi_i; \ \xi_i \ge 0$$

其中 $k(x, x_i)$ 为核函数，$K$ 为核矩阵，$K_i$ 为核矩阵的第 $i$ 行。可以验证，当核函数为内积时，以上问题等价于线性核问题。

## 3. 关于优化方法

国内多数的博客里，都将成本函数转化为对偶问题，在满足KKT优化条件的前提下应用SMO算法。其实，原问题本身就是典型的Quadratic Programming问题。对于编写原型代码来说，没有必要转化成对偶问题计算。本人使用了Python的cvxpy库（由"Convex Optimization"的作者亲自开发），对以上表达式直接优化。

需要注意的是，cvxpy对于优化表达式有独立的一套函数，并不能和Numpy语法完全兼容，对于非Matlab用户来说很难受。建议阅读官网上逻辑回归与SVM的例子，但要注意的是这里的优化表达式与例子中的不一样。

## 4. Python实现

```python
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
```