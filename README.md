
# 常用机器学习算法原型代码 

Self-contained prototype coding for common Machine Learning algorithms in Python 3. 

This repository is created for practice purpose. I'll bring more mathematical brief behind the algorithms.

## Finished

### 协同过滤与SVD推荐

` ml_cf_svd.py`
 
- Collaborate Filtering (for explicit scores)
- SVD approach for recommending, solved with gradient descent (for explicit scores)
- SVD approach for recommending, solved with ALS (for implicit scores)

### 三种经典决策树

`ml_decesion_tree.py`

Training, testing and plot functions for the 3 common DTS algorithms:
- ID3
- C45
- CART

Features:
- Argument `d` in `train_cart()` and `train_id3()` function to control the tree depth.
- Argument `m` and `sample` in `train_cart()` and `train_id3()` function to control the number of randomized feature candidates.
- Missing values in both training and testing dataset will be automotically broadcast into all sub-branches to get weight results.

### 随机森林 

`ml_random_forrest.py`

Random Forrest based on `ml_decision_tree.py`.

### 正则化逻辑回归

`ml_logistic_regression.py`

- basic logistic regression, solved with CVX
- L1 & L2 penalized logistic regression, solved with CVX

### 支持向量机

`ml_svm.py`

SVM solved with CVX for 3 kinds of kernels:
- linear kernel
- gaussian kernel
- polynomial kernel

### AdaBoost

`ml_adaboost.py`

AdaBoost based on `ml_decision_tree.py`. Argument `d` in `train()` function to control tree depth for classifier weakness.

### Gradient Boosting 

`ml_gbdt.py`

Gradient Boosting Decision Tree based on `ml_decision_tree.py`. In fact this is a instance of a broader concept of Gradient Boosting. GBDT takes the loss function of SSE, which is common for regression trees.

### K-Means Clustering

`ml_kmeans.py`

K-Means clustering initializing centroids randomly and stopping when average of centroid movement falls below a threshold.

## To-Do 

### 神经网络

`ml_neural_networks.py`

Neural Networks solved with gradient descent based on Andrew Ng's course.

### SVD++拆分

`ml_cf_svd_plus.py`

Re-arrange ml_cf_svd.py, store CF and SVD methods separately.

### EM

### FP-tree

> side project: data pre-processing toolkit including PCA, standardize, Box-Cox, collinearity removal.
