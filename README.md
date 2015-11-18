# algorithm_ml

Prototype coding for common Machine Learning algorithms. 

This repository is created for practice purpose. I'll bring more mathematical brief behind the algorithms.

## finished
### ml_cf_svd.py 
- Collaborate Filtering (for explicit scores)
- SVD approach for recommending, optimized with gradient descent (for explicit scores)
- SVD approach for recommending, optimized with ALS (for implicit scores)

### ml_decesion_trees.py
All-in-one training, testing and plot functions for the 3 common DTS algorithms:
- ID3
- C45
- CART

Missing values in both training and testing dataset will be handled automotically (broadcasting into all sub-branches to get weight results).

### ml_logistic_regression.py
- basic logistic regression, optimized with CVX
- L1 & L2 penalized logistic regression, optimized with CVX

### ml_svm.py
SVM optimized with CVX for 3 kinds of kernels:
- linear kernel
- gaussian kernel
- polynomial kernel

## to-do
### ml_ann.py
Neurual Network optimized with gradient descent based on Andrew Ng's course.

### ml_random_forrest.py

### ml_boosting.py

### ml_cf_svd_plus.py
