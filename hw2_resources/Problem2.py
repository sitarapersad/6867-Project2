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
'''


def dual_SVM(X, Y, C, K):
	'''
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

	@params:
		X - 
		Y - 
		C - 
		K - 

	@returns:
		alpha - 	
	'''

	# If K is a function, use it to create the Gram matrix

	n, d  = X.shape # n is the number of samples of X, d is the dimension . n is also the length of alpha

	I_d = np.identity(d)
	G = np.vstack((-1*I_d, I_d))

	C_col = np.zeros(n,1)
	C_col.fill(C)
	h = np.vstack( (np.zeros((n,1)), C_col) )

	q  = np.zeros(n,1)
	q.fill(-1)

 	P = matrix(np.outer(Y,Y) * K)
	q = matrix(q)
	G = matrix(G)
	h = matrix(h)
	A = matrix(Y)
	b = matrix(0)

	# Use cvxopt to solve QP
	solve = opt.solvers.qp(P,q,G,h,A,b)

	print solve

	# How to extract alphas from this ; and get non-zero alphas to select SVMS 


	returns TODO
