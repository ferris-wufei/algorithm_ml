# 常用机器学习算法原型代码 

Prototype coding for common Machine Learning algorithms with Python 3. 

This repository is created for personal practice purpose. I'll bring more mathematical brief behind the algorithms.

## Finished
### 协同过滤与SVD推荐
` ml_cf_svd.py`
 
- Collaborate Filtering (for explicit scores)
- SVD approach for recommending, solved with gradient descent (for explicit scores)
- SVD approach for recommending, solved with ALS (for implicit scores)

### 三种经典决策树
`ml_decesion_trees.py`

All-in-one training, testing and plot functions for the 3 common DTS algorithms:
- ID3
- C45
- CART

Features:
- Missing values in both training and testing dataset will be automotically broadcast into all sub-branches to get weight results.
- Support of randomized training feature candidates in function train() to incorporate with Random Forrest algorithm.

### 随机森林 
`ml_random_forrest.py`

Random Forrest based on ml_decision_trees.py

### ml_logistic_regression.py
- basic logistic regression, solved with CVX
- L1 & L2 penalized logistic regression, solved with CVX

### 支持向量机
`ml_svm.py`

SVM solved with CVX for 3 kinds of kernels:
- linear kernel
- gaussian kernel
- polynomial kernel

## To-Do 
### 神经网络
`ml_ann.py`

Neurual Network solved with gradient descent based on Andrew Ng's course.

### 随机梯度加速器
`ml_boosting.py`

Boosting for regression and classification (adaboost)

### SVD++拆分 
`ml_cf_svd_plus.py`
Re-arrange ml_cf_svd.py, store CF and SVD methods separately.
