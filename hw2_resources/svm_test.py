from numpy import *
from plotBoundary import *
import pylab as pl
import Problem2 as p2

# parameters
name = 'ls'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Define parameters
C = 1
K = identity_gram(X)

def column_kernel(SVM_X,x,gamma):
    '''
    Given an array of X values and a new x to predict, 
    computes the  vector whose i^th entry is k(SVM_X[i],x)
    '''
    def k(y):
        return np.dot(x,y) # returns the identity kernel
        # return (1+np.dot(x,y)) # returns the linear basis kernel
        # return # returns the Gaussian RBF function
    return np.apply_along_axis(k, 1, SVM_X ).reshape(-1,1)

# Carry out training, primal and/or dual
a, SVM_a, SVM_X, SVM_Y = dual_SVM(X, Y, C, K)


# Define the predictSVM(x) function, which uses trained parameters
# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
def predict_gaussianSVM(x):
    '''
    The predicted value is given by h(x) = sign( sum_{support vectors} alpha_i y_i k(x_i,x) )
    '''
    debug = False

    x = x.reshape(1, -1)
    if debug: print 'Classify x: ',x
    ay = SVM_alpha*SVM_Y

    # predict new Y output
    kernel = column_kernel(SVM_X, x, gamma)
    y = np.dot(ay.T, kernel)
    if debug:
        print 'New y ', y
    return y

def classification_error(X_train, Y_train):
    ''' Computes the error of the classifier on some training set'''
    n,d = X_train.shape
    incorrect = 0.0
    for i in range(len(n)):
        if predict_SVM(X_train[i]) * Y_train[i] < 0:
            incorrect += 1
    return incorrect/n
    

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
