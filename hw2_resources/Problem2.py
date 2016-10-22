# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 00:52:43 2016

@author: Sitara
"""

#PROBLEM 1: Logistic Regression (LR)
import sklearn.linear_model as linear

#PART A: Gradient Descent
'''
 Use a gradient descent method to optimize the logistic regression objective,
 with L2 regularization on the weight vector
 '''
 
 # Can we use the modified gradient descent we 
 # made in HW1
 def gradient_descent(data, L, max_epochs):
     pass
 
 
#PART B: Comparing Regularization
'''
Let’s now compare the effects of L1 and L2 regularization on LR. 
Minimize NLL + Lambda(Lx Norm).

Evaluate the effect of the choice of regularizer (L1 vs L2) and 
the value of λ on (a) the weights, (b) the decision boundary and (c)
 the classiﬁcation error rate in each of the training data sets


QUESTION: Doesn't the weight define the decision boundary what does it MEANNNNNN

sklearn: 
LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
                   fit_intercept=True, intercept_scaling=1, 
                   class_weight=None, random_state=None, solver='liblinear', 
                   max_iter=100, multi_class='ovr', verbose=0, 
                   warm_start=False, n_jobs=1)
                   
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
'''

def compare_logit(data, L):
    ''' Accepts a dataset and lambda value, L.
        Computes the weight vectors in the case of L1 & L2 regression.
        
        Plots the weight values for L1 and L2 for a given L.
        Outputs the classification error in the L1 and L2 case.
        
    '''
    
    
    