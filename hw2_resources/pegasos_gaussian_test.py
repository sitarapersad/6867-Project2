from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np

# import your LR training code


def gaussian_gram(x, sigma):
	'''Given a dataset and bandwidth sigma, computes the Gaussian RBF kernel matrix'''

	# recast x matrix as floats
	x = np.asfarray(x)

	# get a matrix where the (i, j)th element is |x[i] - x[j]|^2
	pt_sq_norms = (x ** 2).sum(axis=1)
	dists_sq = np.dot(x, x.T)
	dists_sq *= -2.0
	dists_sq += pt_sq_norms.reshape(-1, 1)
	dists_sq += pt_sq_norms
	# turn into an RBF gram matrix
	km = dists_sq; del dists_sq
	km *= -1./(2 * sigma)
	K = np.exp(km)  # exponentiates in-place
	return K 

def train_gaussianSVM(X, Y, K, L, max_epochs):
	debug = False
	t = 0
	assert L != 0; 'Lambda must be non-zero'
	# Initialise w
	if debug: 
		print 'X data', X.shape
		print 'Y data', Y.shape

	n, d = X.shape # n = number of data points, d = dimension

	nY, dY = Y.shape # n = number of points, c should be 1

	assert n == nY; 'X and Y should have same number of sammples'
	alpha = np.empty((n, 1))	# a is the vector of alphas of dimension n x 1

	if debug: print 'Initial alpha matrix: ',alpha.shape

	epoch = 0
	while epoch < max_epochs:
		for i in range(n):
			t += 1
			eta = 1.0/(t*L)
			K_col = K[:,i]   #grab relevant column of K
			S = np.dot(alpha.T, K_col)   #compute Y[i] * sum_j a_j K(x_j, x_i)
			if Y[i] * S < 1:
				alpha[i] = (1-eta*L)*alpha[i] + eta*Y[i]
			else:
				alpha[i] = (1-eta*L)*alpha[i]
		epoch += 1

	#grab support vectors at alpha = 0
	support = alpha > 1e-3 # or alpha<-1e-3

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


# load data from csv files
train = loadtxt('data/data1_train.csv')
X = train[:,0:2]
Y = train[:,2:3]


X = np.asfarray([[2,2],[2,3],[0,-1],[-3,-2]])
Y = np.asfarray([[1],[1],[-1],[-1]])
# Carry out training.
epochs = 1000;
lmbda = .02;
global gamma 
gamma = 0.5;


K = gaussian_gram(X, gamma)

global alpha, SVM_alpha, SVM_X, SVM_Y
alpha, SVM_alpha, SVM_X, SVM_Y = train_gaussianSVM(X, Y, K, lmbda, epochs);


print SVM_X
def gaussian_kernel(SVM_X,x,gamma):
	'''
	Given an array of X values and a new x to predict, 
	computes the  vector whose i^th entry is k(SVM_X[i],x)
	'''

	def gauss(y):
		sqr_diff = np.linalg.norm(x-y)**2
		sqr_diff *= -1./(2)
		return np.exp(sqr_diff)

	return np.apply_along_axis(gauss, 1, SVM_X ).reshape(-1,1)

# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
def predict_gaussianSVM(x):
	'''
	The predicted value is given by h(x) = sign( sum_{support vectors} alpha_i y_i x_i.T * x )
	'''
	debug = False

	x = x.reshape(1, -1)
	if debug: print 'Classify x: ',x
	# compute w:
	ay = SVM_alpha*SVM_Y
	w = np.dot(ay.T, SVM_X)

	# # compute b??
	# b = SVM_Y[0] - np.dot(SVM_X[0],w.T)

	# predict new Y output
	kernel = gaussian_kernel(SVM_X, x, gamma)

	y = np.dot(ay.T, kernel)

	if debug:
		print 'New y ', y

	return y


# plot training results
plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
pl.show()
