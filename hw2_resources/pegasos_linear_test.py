from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
import Problem3 as p3


name = '1'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
L = 2e-5
max_epochs = 1000



global w
w = p3.train_linearSVM(X, Y, L, max_epochs)

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

