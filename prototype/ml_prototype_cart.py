# -*- coding: utf-8 -*-
"""Module docstring.

参考: "集体智慧编程"中的CART算法

To Do
1. (Done) python3兼容性修改
2. (Done) 部分代码错误修正
3. 新增回归树的生长, 剪枝和预测(空值处理)函数
4. 在原有函数基础上实现习题中的功能
5. OOP + 合并
6. 学习C45算法

"""

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


# 定义递归式节点对象, 表示树
class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        # 终端节点只需要results属性, 非终端节点只需要其他属性
        self.results = results
        # 递归式对象,每个非终端节点对应两个子节点
        self.tb = tb
        self.fb = fb


# CART 二叉树划分函数
def divideset(rows, column, value):
    # split_function = None
    # if isinstance(value, int) or isinstance(value, float):
    #     split_function = lambda row: row[column] >= value
    # else:
    #     split_function = lambda row: row[column] == value
    if isinstance(value, int) or isinstance(value, float):
        def split_function(row):
            return row[column] >= value
    else:
        def split_function(row):
            return row[column] == value
    # 非空值, 按照value拆分; 空值, 同时发送到2个list
    set1 = [row for row in rows if split_function(row) or row[column] is None]
    set2 = [row for row in rows if not split_function(row) or row[column] is None]
    return set1, set2


# 对数据集rows的最后一列的值生成计数字典
def uniquecounts(rows):
    results = {}
    for row in rows:
        # 默认结果在数集rows的最后一列
        r = row[len(row) - 1]
        results[r] = results.get(r, 0) + 1
    return results


# 按照最后一列计算基尼系数
def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


# 按照最后一列计算熵
def entropy(rows):
    from math import log
    results = uniquecounts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log(p)
    return ent


# 递归建立树模型
def buildtree(rows, scoref=giniimpurity, stop=0):
    if len(rows) == 0:
        return decisionnode()

    current_score = scoref(rows)

    # 创建追踪器
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    # 排除结果列(最后一列)
    column_count = len(rows[0]) - 1
    for col in range(column_count):
        column_values = {}
        # 获取该列的所有唯一值,
        for row in rows:
            column_values[row[col]] = 1
        for value in column_values.keys():
            # None值不能用于划分
            if value is None:
                continue
            (set1, set2) = divideset(rows, col, value)
            p = float(len(set1)) / len(rows)
            # 计算entropy减量
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                # 更新3个追踪器, 用于其他与其他拆分条件的减量比较
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # 递归创建子节点. stop控制entropy减量的最低条件
    if best_gain > stop:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        # 返回当前非终端节点
        return decisionnode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        # 返回终端节点
        return decisionnode(results=uniquecounts(rows))


# 以文本形式递归显式树
def printtree(tree, indent=''):
    # 判断是否终端节点: 如果是, 显式计数字典
    if tree.results is not None:
        print(str(tree.results))
    # 分支节点
    else:
        print(str(tree.col) + ':' + str(tree.value) + '? ')
        # 递归显示分支节点
        print(indent + 'T->', end='')
        printtree(tree.tb, indent + '  ')
        print(indent + 'F->', end='')
        printtree(tree.fb, indent + '  ')


# 剪枝: 先完全成长, 再从终端节点开始合并"entropy减量 < mingain"的节点
def prune(tree, mingain):
    # 子节点为分支节点, 递归执行
    if tree.tb.results is None:
        prune(tree.tb, mingain)
    if tree.fb.results is None:
        prune(tree.fb, mingain)
    # 子节点为终端节点, 执行剪枝
    if tree.tb.results is not None and tree.fb.results is not None:
        tb = []
        fb = []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c
        delta = entropy(tb + fb) - ((entropy(tb) + entropy(fb)) / 2)
        # entropy减量满足阈值范围, 删除子节点, 当前节点改为终端节点
        if delta < mingain:
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)

# ---------------------------------------------------------------------------
# 预测判别的空值处理: 在空值的条件判断节点, 分别按两个分支计算最终结果.
# 将两个分支中, 样本归属的终端节点数据, 按照终端节点的数据量加权求和, 作为实际终端节点数据.
# ---------------------------------------------------------------------------


# 预测判别函数, 返回observation属于各个分类的概率
def mdclassify(observation, tree):
    # 到达终端节点, 返回计数字典
    if tree.results is not None:
        result = tree.results
        for k in result:
            result[k] = result[k] / sum(result.values())
        return result
    # 未到达终端节点, 递归处理
    else:
        v = observation[tree.col]
        # 空值处理: 同时计算2条branch的结果, 对结果的计数加权求和
        if v is None:
            tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in tr.items():
                result[k] = v * tw
            for k, v in fr.items():  # 原书有误. 应使用累加避免不同节点内相同key对应的值发生覆盖
                if k in result:
                    result[k] += v * fw
                else:
                    result[k] = v * fw
            for k in result:
                result[k] = result[k] / sum(result.values())
            return result
        # 非空值: 选择正确的branch, 递归处理
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return mdclassify(observation, branch)


# 计算数据集rows最后一列值的方差
def variance(rows):
    if len(rows) == 0:
        return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    var = sum([(d - mean) ** 2 for d in data]) / len(data)
    return var


# 回归树对应的成长, 剪枝和预测函数都需要重写, 或者对原函数添加判断
