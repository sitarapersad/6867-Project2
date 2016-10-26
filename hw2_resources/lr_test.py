from numpy import *
from plotBoundary import *
import pylab as pl
# import your LR training code
import numpy as np
from sklearn.linear_model import LogisticRegression
import csv

# Carry out training.
def getLogisticRegression(regularization_type, inverse_lambda, iter):
  if regularization_type == 1:
    return LogisticRegression(penalty = "l1", C = inverse_lambda, max_iter = iter)
  else:
    return LogisticRegression(penalty = "l2", C = inverse_lambda, max_iter = iter)

# parameters
# names = [str(i) for i in xrange(1, 5)]
def model_selection():
  names = ['1', '2', '3', '4']
  with open('updated_results.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter = ' ')
    for name in names:
      # load data from csv files
      train = loadtxt('data/data'+name+'_train.csv')
      Xtrain = train[:,0:2]
      Ytrain = train[:,2:3]
      validate = loadtxt('data/data'+name+'_validate.csv')
      Xvalidate = validate[:,0:2]
      Yvalidate = validate[:,2:3]
      test = loadtxt('data/data'+name+'_test.csv')
      Xtest = test[:,0:2]
      Ytest = test[:,2:3]
      spamwriter.writerow(name)

      inverse_lambdas = [1e-2, 1e-1, 1/5.0, 1/2.0, 1, 2, 5, 10, 10e2, 10e50]
      results = []
      for i in range(2):
        for j in inverse_lambdas:
          lr = getLogisticRegression(i + 1, j, 5000)
          print '======Training======'
          lr.fit(Xtrain, Ytrain)
          def predictLR(x):
            return lr.predict_proba(np.array([x]))[0][1]
          # plotDecisionBoundary(Xtrain, Ytrain, predictLR, [0.5], title = 'LR Train')
          pl.show()
          print '======Validation======'
          results.append([i + 1, 1.0/j, lr.coef_, np.linalg.norm(lr.coef_), 1 - lr.score(Xvalidate, Yvalidate), 1 - lr.score(Xtest, Ytest)])
          # plotDecisionBoundary(Xvalidate, Yvalidate, predictLR, [0.5], title = 'LR Validate')
          pl.show()
          print '======Testing======'
      results.sort(key = lambda x: x[-2])
      for result in results:
        spamwriter.writerow(result)

          # print name, i + 1, 1.0/j, lr.coef_, np.linalg.norm(lr.coef_), lr.score(X, Y)

# # Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
  return lr.predict_proba(np.array([x]))[0][1]

def classificatonError(lr, X, Y):
  return lr.score(X, Y)

<<<<<<< HEAD
model_selection()

# # plot training results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')
=======
 # plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')
pl.show()
>>>>>>> 674832a5662abaacad4427f9cc9c22d71ad9cca8

print '======Validation======'
 # load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# # plot validation results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
# pl.show()
