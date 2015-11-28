# Machine Learning Algorithms

Self-contained prototype coding for common Machine Learning algorithms in Python 3. 

This repository is created for practice purpose and written in pure Python. Alhough I did with Numpy as much as possible, speed performance should be considered if you want to implement on big data.

In addition to the code comments, I've kept Markdown notes with mathematical details of all the algorithms here. Please email me at `wu.fei@outlook.com` if you need any of them.

## Finished

### Collaborative Filtering and SVD++

` ml_cf_svd.py`
 
- Collaborate Filtering (for explicit scores)
- SVD approach for recommending, solved with gradient descent (for explicit scores)
- SVD approach for recommending, solved with ALS (for implicit scores)

### Decision Tree

`ml_decesion_tree.py`

Training, testing and plot functions for the 3 common DTS algorithms:
- ID3
- C45
- CART

Features:
- Argument `d` in `train_cart()` and `train_id3()` function to control the tree depth.
- Argument `m` and `sample` in `train_cart()` and `train_id3()` function to control the number of randomized feature candidates.
- Missing values in both training and testing dataset will be automotically broadcast into all sub-branches to get weight results.

### Random Forrest

`ml_random_forrest.py`

Random Forrest based on `ml_decision_tree.py`.

### Regularized Logistic Regression

`ml_logistic_regression.py`

- basic logistic regression, solved with CVX
- L1 & L2 penalized logistic regression, solved with CVX

### Support Vector Machine

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

### Gaussian Mixture Model

`ml_gmm.py`

Gaussian Mixture Model solved with EM algorithm. The initializing setup is to assign samples into k populations, and then initialize the gaussian distribution parameters with the estimate of each population. The stopping criteria is average of Euclidean distance between new and old mu vector below some threshold.

## To-Do 

### 神经网络

`ml_neural_networks.py`

Neural Networks solved with gradient descent based on Andrew Ng's course.

### SVD++拆分

`ml_cf_svd_plus.py`

Re-arrange ml_cf_svd.py, store CF and SVD methods separately.

### FP-tree

> side project: data pre-processing toolkit including PCA, standardize, Box-Cox, collinearity removal.
