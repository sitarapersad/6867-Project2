from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
#import Problem3 as p3

def train_linearSVM(X, Y, L, max_epochs):
	'''
	Matrices of the form:

	w = [w0 w1 ... w_d+1]

	X = [[1 X[1]_1 ... X[1]_d]
         [1 X[2]_1 ... X[2]_d]
		  :   :          :
         [1 X[n]_1 ... X[n]_d]]

	'''
	debug = False 
	t = 0
	assert L != 0; 'Lambda must be non-zero'

	n, d = X.shape # n = number of data points, d = dimension

	# Add bias term to linear SVM; row of 1's as first dimension of X
	X = np.hstack((np.ones((n,1)),X))

	# Initialise w
	if debug: 
		print 'X data', X
		print 'Y data', Y

	n, d = X.shape # n = number of data points, d = dimension
	nY, dY = Y.shape # n = number of points, c should be 1

	w = np.zeros((1, d)) # w is the (horizontal) weight vector of dimension 1 x d

	assert n == nY; 'X and Y should have same number of sammples'

	if debug: print 'weight matrix: ',w

	epoch = 0
	while epoch < max_epochs:
		for i in range(n):
			t += 1
			eta = 1.0/(t*L)
			unbiased_w = np.copy(w)
			unbiased_w[0][0] = 0
			if Y[i] * np.dot(w, X[i].T) < 1:
				w = w - eta*L*unbiased_w + eta*Y[i]*X[i]
			else:
				w = w - eta*L*unbiased_w 
		epoch += 1

	return w

name = '1'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
L = 2e-5
max_epochs = 1000

X = np.asfarray([[2,2],[2,3],[0,-1],[-3,-2]])
Y = np.asfarray([[1],[1],[-1],[-1]])

global w
w = train_linearSVM(X, Y, L, max_epochs)

print '===WEIGHT VECTORS FOR DATA SET ', name, ' WITH L = ', L, ' ===='
print w 
print ''


# Define the predict_linearSVM(x) function, which uses global trained parameters, w
def predict_linearSVM(x):
	'''
	Given a data set, returns a vector of predictions
	using the trained parameters w
	'''
	debug = False
	try:
		n, dim_x = x.shape
	except ValueError:
		dim_x = x.shape[0]
		x = np.reshape(x, (1,dim_x))
		n, dim_x = x.shape

	one, dim_w = w.shape
	assert one == 1; 'w is not a row vector'

	if dim_x == dim_w:
		pass
	elif dim_x +1 == dim_w:
		# add a 1 to the first term of each data point to account for w bias term
		x = np.hstack((np.ones((n,1)),x))
	else:
		assert dim_x == dim_w, 'X and w have incompatible dimensions'
	if debug: print x, w  

	# the prediction depends on the sign of w*x.T; y is row vector of predictions 
	Y = np.dot(w,x.T)[0][0]
	
	if debug: print Y
	
	return Y


# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()

