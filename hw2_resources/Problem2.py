#PROBLEM 2: Logistic Regression (LR)

import numpy as np
import cvxopt as opt 

#PART A: Dual Form of Linear SVMs
'''
Implement the dual form of linear SVMs with slack variables. Please do not use the built-in SVM 
implementation in Matlab or sklearn. Instead, write a program that takes data as input, converts
it to the appropriate objective function and constraints, and then calls a quadratic programming
package to solve it. See the file optimizers.txt for installation and usage for matlab/python.
'''

# JUST IMPLEMENT KERNEL VERSION AND USE phi(x) = x for PART A 

def identity_gram(X):
    '''Given a dataset, computes the gram matrix using phi(x) = x '''
    def k(x,y):
        return np.dot(x,y)
        
    return compute_gram(X,k)

def linear_gram(X, m=1):
    '''Given a dataset, computes the gram matrix using k(x,y) = (1+xy)^m, m=1 '''
    def k(x,y):
        return (1+np.dot(x,y))**m
        
    return compute_gram(X,k)
    
def compute_gram(X, k):
    ''' Given a function k and a datasdet X, computes the Gram matrix
    slow as fudge'''
    n, d = X.shape
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(j):
            K[i,j] = k(X[i], X[j])
    return K

#PART B: Test Your Dual Form of Linear SVMs
'''
Implement the dual form of linear SVMs with slack variables. Please do not use the built-in SVM 
implementation in Matlab or sklearn. Instead, write a program that takes data as input, converts
it to the appropriate objective function and constraints, and then calls a quadratic programming
package to solve it. See the file optimizers.txt for installation and usage for matlab/python.
'''


 
#PART C: Kernelized Dual Form of Linear SVMs
'''
The dual form SVM is useful for several reasons, including an ability to handle kernel functions 
that are hard to express as feature functions in the primal form. Extend your dual form SVM code 
to operate with kernels. Do the implementation as generally as possible, so that it either takes
 the kernel function or kernel matrix as input.

The cvxopt package solves the minimization problem:
    min 1/2 x.T P x + q.T x
    subject to:
        Gx <= h
        Ax =  b

The kernel SVM problem is thus reformulated to match this format:
    x = alpha
    P = outer(Y,Y) * K

    q = col vector of [-1] * len(alpha)

    A = Y, b = 0
    Gx <= h such that 0<=a_i<=C
'''


def dual_SVM(X, Y, C, K):
    '''
    @params:
        X - n x d data matrix
        Y - n x 1 classification vector
        C - 
        K - gram matrix or kernel function
    @returns:
        alpha - set of non-negative weights to be used in classification
    '''
    debug = False

    # If K is a function, use it to create the Gram matrix
    if callable(K):
        K = compute_gram(X,K)

    n, d  = X.shape # n is the number of samples of X, d is the dimension . n is also the length of alpha

    I_d = np.identity(d)
    G = np.vstack((-1*I_d, I_d))

    C_col = np.zeros(n,1)
    C_col.fill(C)
    h = np.vstack( (np.zeros((n,1)), C_col) )

    q  = np.zeros(n,1)
    q.fill(-1)

    P = opt.matrix(np.outer(Y,Y) * K)
    q = opt.matrix(q)
    G = opt.matrix(G)
    h = opt.matrix(h)
    A = opt.matrix(Y)
    b = opt.matrix(0)

    # Use cvxopt to solve QP
    solve = opt.solvers.qp(P,q,G,h,A,b)
    alpha = solve['x']

    # Extract SVMs from alpha vector -  this is where alpha_i > 0 

    support = alpha > 1e-5
    if debug:
        print 'Support', support.shape

    SVM_alpha = alpha[support].reshape(-1,1)
    SVM_X = X[support.flatten()].reshape(-1,d)
    SVM_Y = Y[support].reshape(-1,1)

    if debug:
        print 'alphas', SVM_alpha.shape
        print 'X', SVM_X.shape
        print 'Y', SVM_Y.shape

    return alpha, SVM_alpha, SVM_X, SVM_Y
    