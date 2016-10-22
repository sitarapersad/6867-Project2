#PROBLEM 3: Support Vector Machine with Pegasos


#PART A: PEGASOS 
'''
The following pseudo-code is a slight variation on the Pegasos learning 
algorithm, with a ﬁxed iteration count and non-random presentation of the 
training data. Implement it, and then add a bias term (w0) to the hypothesis, 
but take care not to penalize the magnitude of w0. Your function should output
classiﬁer weights for a linear decision boundary.
'''

def train_linearSVM(X, Y, L, max_epochs):
	debug = False 
	t = 0
	assert L != 0; 'Lambda must be non-zero'
	# Initialise w
	if debug: 
		print 'X data', X
		print 'Y data', Y

	n. d = X.shape # n = number of data points, d = dimension
	nY, dY = Y.shape # n = number of points, c should be 1

	# TO DO: ADD BIAS TERM; DO NOT PENALIZE IT

	assert n == nY; 'X and Y should have same number of sammples'
	w = np.empty([d, 1])	# w is the weight vector of dimension d x 1

	if debug: print 'weight matrix: ',w

	epoch = 0
	while epoch < max_epochs:
		for i in range(n):
			t += 1
			eta = 1.0/(t*L)
			if Y[i] * np.dot(w, X[i]) < 1:
				w = (1-eta*L)*w + eta*Y[i]*X[i]
			else:
				w = (1-eta*L)*w 
		epoch += 1

	return w


#PART B
'''
Test various values of the regularization constant, L = 2 , . . . , 2e−10 . Observe the the margin 
(distance between the decision boundary and margin boundary) as a function of L. 
Does this match your understanding of the objective function?
'''

L_test = [2e-(i+1) for i in range(10)]


#PART C: KERNELIZED SOFT SVM
'''
We can also solve the following kernelized Soft-SVM problem with a few extensions to the above algorithm. 
Rather than maintaining the w vector in the dimensionality of the data, we maintain α coefficients for 
each training instance.

Implement a kernelized version of the Pegasos algorithm. It should take in a Gram matrix, where entry 
i, j is K(x(i), x(j)) = phi(x(i)) * phi(x(j)), and should should output the support vector values, alpha,
or a function that makes a prediction for a new input. In this version, you do not need to add a bias term.
'''

def train_gaussianSVM(X, Y, K, L, max_epochs):
	debug = False
	t = 0
	assert L != 0; 'Lambda must be non-zero'
	# Initialise w
	if debug: 
		print 'X data', X
		print 'Y data', Y

	n. d = X.shape # n = number of data points, d = dimension
	nY, dY = Y.shape # n = number of points, c should be 1

	# TO DO: ADD BIAS TERM; DO NOT PENALIZE IT

	assert n == nY; 'X and Y should have same number of sammples'
	alpha = np.empty([d, 1])	# a is the vector of alphas of dimension d x 1

	if debug: print 'Initial alpha matrix: ',a

	epoch = 0
	while epoch < max_epochs:
		for i in range(n):
			t += 1
			eta = 1.0/(t*L)

			# Compute Y[i] * sum_j a_j K(x_j, x_i)

			#grab relevant column of K
			K_col = K[:,i]
			S = np.dot(alpha.T, K_col)
			if Y[i] * S < 1:
				alpha[i] = (1-eta*L)*alpha[i] + eta*Y[i]
			else:
				alpha[i] = (1-eta*L)*alpha[i]
		epoch += 1

	return alpha
