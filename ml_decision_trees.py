# -*- coding: utf-8 -*-
"""Module docstring.

@Author: Ferris

说明: 在同一组函数实现ID3，C45，CART三种算法的分类与回归树，实现同时处理二分支和多分支节点，连续和离散变量，并自动处理空值.

函数功能:
1. train 生长
2. predict 预测
3. prune 剪枝
4. plottree 绘图

假设: 数据集的最后一列为结果。

To-Do:
1. (Done)plot函数
2. (Done)predict函数
3. (Done)prune函数(剪枝)
4. (Done)bug: 如果rows只有2个记录，且唯一的差别是当中一条的某个字段为空，一条非空，按照这个字段划分时, 会导致死循环

"""
import numpy as np

# sample data
my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]


# 定义节点, 同时支持ID3的多分和CART的二分, 作为所有决策树的组成对象
class Node:
    def __init__(self, dataset=None, feature=None, algo='cart',
                 cut=None, tb=None, fb=None, children={}, target='classification'):
        self.algo = algo  # id3, c45, cart
        self.target = target  # 在train的时候进行赋值, 带到predict和plot
        self.feature = feature  # 用于划分的特征index
        self.cut = cut  # 二分分解点, 这个参数用于在预测是判断拆分类型
        self.dataset = dataset  # 只有终端节点才有数据
        self.children = children  # 多分支无序节点
        self.tb = tb  # 二分True有序节点
        self.fb = fb  # 二分False有序节点


# 字典生成: key为y标签, value为样本数量
def countgen(rows):
    dic = {}
    for row in rows:
        r = len(row) - 1  # 结果在最后一列
        dic[row[r]] = dic.get(row[r], 0) + 1
    return dic


# 返回字典中value最大的key值, 用于分类树的predict
def topkey(dict):
    key_tracker = None
    value_tracker = 0.0
    for k, v in dict.items():
        if v > value_tracker:
            key_tracker = k
            value_tracker = v
    return key_tracker


# 分类树：计算熵
def entropy(rows, base=2):
    from math import log
    results = countgen(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log(p) / log(base)
    return ent


# 分类树：计算基尼系数
def gini(rows):
    results = countgen(rows)
    gini = 1
    for r in results.keys():
        gini -= (float(results[r]) / len(rows)) ** 2
    return gini


# 回归树：计算误差平方和
def rss(rows):
    data = [row[len(row) - 1] for row in rows]
    mean = sum(data) / len(data)
    rs = sum([(row[len(row) - 1] - mean) ** 2 for row in rows])
    return rs


# 离散和连续变量: 二分函数
def divide2(rows, feature, cut):
    if isinstance(cut, float) or isinstance(cut, int):  # 判断离散或连续变量
        def split_bool(row):
            return row[feature] >= cut
    else:
        def split_bool(row):
            return row[feature] == cut
    # 空值样本同时发送到2个分支
    t_set = [row for row in rows if split_bool(row) or row[feature] is None]
    f_set = [row for row in rows if not split_bool(row) or row[feature] is None]
    return t_set, f_set


# 离散变量: 多分函数
def divide3(rows, feature):
    if not isinstance(rows, (set, list)):  # 参数类型异常处理
        return None
    subset = {}
    for row in rows:
        subset.setdefault(row[feature], []).append(row)
    return subset


# 训练函数
def train(rows, threshold=0.0, algo="cart", target="classification",
          m=2, sample=False):
    """
    递归生长思路:
    1. 只要当前数据满足继续划分的条件, 即将划分的数据递归转移到下层节点继续划分,
    并在当前节点的children或tb/fb属性里记录从属关系; 否则, 将数据留在当前节点,作为终端节点返回.
    2. 每次划分的关键检查步骤: 当前数据的结果变量是否唯一, 如果唯一则没有必要划分; 当前特征的
    取值是否唯一, 如果唯一则无法继续划分.
    3. 终端节点直接返回数据集, 不计算结果
    4. sample选项: 每次划分时, 从所有特征中随机选取m个特征的子集, 作为候选特征, 用于随机森林

    :param rows: 数据集
    :param threshold: 信息增益或RSS减少量阈值
    :param algo: 算法类型
    :param target: 回归 / 判别
    :param m: 限定遍历的随机特征数
    :param sample: 是否抽样
    :return:
    """
    if not isinstance(rows, (set, list)):  # 参数异常处理
        return None
    if len(rows) == 0:
        return Node()
    num_features = len(rows[0]) - 1
    if m > num_features:
        return None

    if len(countgen(rows)) <= 1:  # rows只有一种结果, 返回终端节点
        return Node(dataset=rows, target=target)

    legal_param = [('cart', 'classification'),
                   ('cart', 'regression'),  # 回归树仅应用于cart算法
                   ('id3', 'classification'),
                   ('c45', 'classification')]
    if (algo, target) not in legal_param:
        print('illegal algorithm or target values')

    best_diff = 0.0  # 初始化追踪器
    best_feature = None
    best_sets = None
    best_cut = 0.0

    if not sample:
        loop_index = range(num_features)
    else:
        indices = np.random.choice(np.arange(num_features), size=m, replace=False)
        loop_index = [int(i) for i in indices]  # 随机选取m个特征

    if algo == 'cart':  # cart划分
        for f in loop_index:  # 遍历特征
            f_values = {}
            for row in rows:  # 获取特征的取值
                if row[f] is not None:  # 空值不用于划分
                    f_values[row[f]] = 1
            if len(f_values) <= 1:  # 忽略取值唯一的特征
                continue
            for value in f_values.keys():  # 遍历取值
                t_set, f_set = divide2(rows, f, value)
                if len(t_set) == 0 or len(f_set) == 0:
                    continue  # 连续变量划分异常处理
                if target == 'classification':  # 判别: 基尼系数
                    p = float(len(t_set)) / len(rows)
                    diff = gini(rows) - p * gini(t_set) - (1 - p) * gini(f_set)
                else:  # 回归: 误差平方和
                    diff = rss(rows) - rss(t_set) - rss(f_set)
                if diff > best_diff:  # 出现新的最佳划分, 更新追踪器
                    best_diff = diff
                    best_feature = f
                    best_cut = value
                    best_sets = (t_set, f_set)

        if best_diff >= threshold and best_sets is not None:  # 应用最佳划分
            t_branch = train(best_sets[0], threshold=threshold, algo=algo, target=target, m=m, sample=sample)
            f_branch = train(best_sets[1], threshold=threshold, algo=algo, target=target, m=m, sample=sample)
            return Node(feature=best_feature, algo=algo, cut=best_cut, tb=t_branch, fb=f_branch, target=target)
        else:
            return Node(dataset=rows, target=target)  # 未找到feature, 或diff未达到划分标准

    else:  # id3 & c45 划分, 只做分类树, 但要判断变量是否离散
        for f in loop_index:  # 遍历特征
            if isinstance(rows[0][f], int) or isinstance(rows[0][f], float):  # 连续变量, 二分
                f_values = {}
                for row in rows:  # 获取特征的取值
                    if row[f] is not None:  # 空值不用于划分
                        f_values[row[f]] = 1
                if len(f_values) <= 1:  # 忽略取值唯一的特征
                    continue
                for value in f_values.keys():  # 遍历取值
                    t_set, f_set = divide2(rows, f, value)
                    if len(t_set) ==0 or len(f_set) == 0:
                        continue
                    if algo == 'id3':
                        diff = entropy(rows) - entropy(t_set) * len(t_set) / len(rows) -\
                               entropy(f_set) * len(f_set) / len(rows)
                    else:
                        base = 0.0
                        for s in (t_set, f_set):
                            if len(s) > 0:
                                base -= len(s) / len(rows) * log(len(s) / len(rows))
                        diff = entropy(rows) - entropy(t_set) * len(t_set) / len(rows) -\
                               entropy(f_set) * len(f_set) / len(rows)
                        diff = diff / base
                    if diff > best_diff and len(t_set) > 0 and len(f_set) > 0:  # 出现新的最佳划分, 更新追踪器
                        best_diff = diff
                        best_feature = f
                        best_cut = value
                        best_sets = (t_set, f_set)

            else:  # 离散变量, 多分
                f_values = {}
                for row in rows:  # 获取特征的取值
                    f_values[row[f]] = 1
                if len(f_values) <= 1:  # 忽略取值唯一的特征
                    continue
                subsets = divide3(rows, f)  # 多分返回字典
                current_score = entropy(rows)
                sub_score = 0.0
                for s in subsets.keys():  # 累加熵
                    sub_score += entropy(subsets[s]) * len(subsets[s]) / len(rows)
                if algo == 'id3':  # id3算法采用信息增益
                    diff = current_score - sub_score  # 计算信息增益
                else:  # c45算法使用信息增益与base的比例, 详见算法
                    from math import log
                    base = 0
                    for s in subsets.keys():  # 计算base
                        base -= len(subsets[s]) / len(rows) * log(len(subsets[s]) / len(rows))
                    diff = (current_score - sub_score) / base
                if diff > best_diff:  # 出现新的最佳划分, 更新追踪器
                    best_diff = diff
                    best_feature = f
                    best_sets = subsets

        if best_diff >= threshold:  # 使用最佳特征和划分点进行划分
            if isinstance(rows[0][best_feature], int) or isinstance(rows[0][best_feature], float):  # 连续变量递归
                t_branch = train(best_sets[0], threshold=threshold, algo=algo, target=target, m=m, sample=sample)
                f_branch = train(best_sets[1], threshold=threshold, algo=algo, target=target, m=m, sample=sample)
                return Node(feature=best_feature, algo=algo, cut=best_cut, tb=t_branch, fb=f_branch,
                            target=target)
            else:  # 离散变量递归
                sub_trees = {}
                for k in best_sets.keys():
                    sub_trees.setdefault(k, train(best_sets[k], threshold=threshold, algo=algo,
                                                  target=target, m=m, sample=sample))
                return Node(feature=best_feature, algo=algo, children=sub_trees, target=target)

        else:
            return Node(dataset=rows)  # 未找到feature, 或diff未达到划分标准


# 递归预测函数
def predict(tree, row, out='value'):
    if tree.dataset is not None:  # 已经到达终端节点
        if out == 'raw':  # 返回原始数据(在ensemble方法中会用到)
            return tree.dataset  # 注意: 当预测样本有空值时, 返回的raw数据包含多个不同分支的乘数增量数据
        else:  # 返回判断结果
            if tree.target == 'classification':  # 分类树: 返回最多结果分类
                return topkey(countgen(tree.dataset))
            else:  # 回归树: 返回结果均值
                return sum([row[len(row) - 1] for row in tree.dataset]) / len(tree.dataset)

    if tree.cut is None:  # 多分支节点
        if row[tree.feature] in tree.children.keys():  # 样本feature在训练集范围(已排除空值情况)
            return predict(tree.children[row[tree.feature]], row, out=out)  # 递归predict子节点
        else:  # 样本feature不在训练集, 等同于空值处理, 同时发送到所有分支, 并在必要时对结果加权处理
            comb = []
            for k in tree.children.keys():
                temp_tree = predict(tree.children[k], row, out='raw')
                comb.extend(temp_tree * len(temp_tree))
            if out == 'raw':
                return comb
            else:
                if tree.target == 'classification':  # 分类树: 返回最多结果分类
                    return topkey(countgen(comb))
                else:  # 回归树: 返回结果均值
                    return sum([row[len(row) - 1] for row in comb]) / len(tree.dataset)
    else:  # 二分支节点
        if row[tree.feature] is not None:  # 样本feature值非空, 递归处理
            if isinstance(tree.cut, int) or isinstance(tree.cut, float):  # 连续变量
                if row[tree.feature] >= tree.cut:
                    return predict(tree.tb, row, out=out)
                else:
                    return predict(tree.fb, row, out=out)
            else:  # 离散变量
                if row[tree.feature] == tree.cut:
                    return predict(tree.tb, row, out=out)
                else:
                    return predict(tree.fb, row, out=out)
        else:  # 样本feature值为空, 将raw结果按照分支长度复制如comb, 再根据out参数返回
            comb = []
            temp_tree = predict(tree.tb, row, out='raw')
            comb.extend(temp_tree * len(temp_tree))
            temp_tree = predict(tree.fb, row, out='raw')
            comb.extend(temp_tree * len(temp_tree))
            if out == 'raw':
                return comb
            else:
                if tree.target == 'classification':  # 分类树: 返回最多结果分类
                    return topkey(countgen(comb))
                else:  # 回归树: 返回结果均值
                    return sum([row[len(row) - 1] for row in comb]) / len(tree.dataset)


# 递归文本树图
def plottree(tree, indent=' '):
    if tree.dataset is not None:  # 终端节点
        if tree.target == 'classification':
            print(topkey(countgen(tree.dataset)))  # 分类树: 返回最大分类
        else:
            print(sum(tree.datasets) / len(tree.dataset))  # 回归树: 返回结果均值
    else:  # 分支节点
        if tree.cut is not None:  # 二分支节点
            print(str(tree.feature) + ':' + str(tree.cut) + '? ')
            print(indent + 'T->', end='')
            plottree(tree.tb, indent + '  ')
            print(indent + 'F->', end='')
            plottree(tree.fb, indent + '  ')
        else:  # 多分支节点
            print(str(tree.feature) + ':' + 'split')
            for k in tree.children.keys():
                print(indent + str(k) + '->', end='')
                plottree(tree.children[k], indent + '  ')


# 递归剪枝
def prune(tree, threshold=0.0):
    if tree.dataset is not None:  # Node类没有定义父节点, 只能从上到下, 不能从终端节点剪枝
        return None

    subtrees = []  # 获取所有子Node对象
    if tree.tb is not None:  # 二分支节点
        subtrees = [tree.tb, tree.fb]
    if len(tree.children) != 0:  # 多分支节点
        subtrees = [v for k, v in tree.children.items()]

    toggle = 0  # 只要有一个非终端子节点, toggle会赋值为1
    for s in subtrees:
        if s.dataset is None:
            prune(s, threshold=threshold)  # 非终端子节点, 函数递归应用到子节点
            toggle = 1

    if toggle == 0:  # 所有子节点都是终端节点, 开始剪枝
        flat_set = []  # 所有子节点的数据合并列表
        for s in subtrees:
            flat_set.extend(s.dataset)
        if tree.algo == 'cart':  # 根据算法获取信息增益
            if tree.target == 'classification':
                diff = gini(flat_set) - sum([gini(s.dataset) for s in subtrees])
            else:
                diff = rss(flat_set) - sum([rss(s.dataset) for s in subtrees])
        elif tree.algo == 'id3':
            diff = entropy(flat_set) - sum([entropy(s.dataset) for s in subtrees])
        else:
            diff = entropy(flat_set) - sum([entropy(s.dataset) for s in subtrees])
            base = 0.0
            for s in subtrees:
                base -= len(s.dataset) / len(flat_set) * np.log(len(s.dataset) / len(flat_set))
            diff = diff / base

        if diff < threshold:  # 信息增益较小, 实行剪枝
            tree.tb, tree.fb, tree.children, tree.cut, tree.features = None, None, None, None, None
            tree.dataset = flat_set
        else:  # 否则停止剪枝
            return None
