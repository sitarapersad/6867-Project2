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

# Define paramters
C = 
K = identity_gram(X)

# Carry out training, primal and/or dual
a, SVM_a, SVM_X, SVM_Y = dual_SVM(X, Y, C, K)


# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
    

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
