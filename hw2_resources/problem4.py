import numpy as np
from plotBoundary import *
import pylab as pl

from lr_test import *
from svm_test_linearKernel_copy import * # C
from svm_test_gaussianKernel_copy import * # C, gamma
from pegasos_gaussian_test_copy import * #lambda = 1/nC, gamma

# from pegasos_linear_test import *
# from pegasos_gaussian_test import *

# e.g., class_digits = [[1], [7]], class_digits = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
def getDataSets(class_digits, normalized):
  data = {}
  handles = ["Xtrain", "Ytrain", "Xvalidate", "Yvalidate", "Xtest", "Ytest"]
  for handle in handles:
    data[handle] = []
  for label in [-1, 1]:
    for digit in class_digits[(label + 1)/2]:
      digits = loadtxt("data/mnist_digit_"+str(digit)+".csv")[0:500, :]
      if normalized:
        digits = 2.0 * np.array(digits) / 255 - 1
      data["Xtrain"].extend(digits[0:200,:])
      data["Ytrain"].extend([label for i in xrange(200)])
      data["Xvalidate"].extend(digits[200:350,:])
      data["Yvalidate"].extend([label for i in xrange(150)])
      data["Xtest"].extend(digits[350:500,:])
      data["Ytest"].extend([label for i in xrange(150)])
  for handle in handles:
    data[handle] = np.array(data[handle])
  return data

data_1vs7 = getDataSets([[1], [8]], False)
# data_1vs7_normalized = getDataSets([[1], [7]], True)
# data_3vs5 = getDataSets([[3], [5]], False)
# data_3vs5_normalized = getDataSets([[3], [5]], True)
# data_4vs9 = getDataSets([[4], [9]], False)
# data_4vs9_normalized = getDataSets([[4], [9]], True)
# data_evenvsodd = getDataSets([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], False)
# data_evenvsodd_normalized = getDataSets([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], True)

def lrModelSelection(data):
  norm = [1, 2]
  inverse_lambdas = [1e-2, 1e-1, 1/5.0, 1/2.0, 1, 2, 5, 10, 10e2, 10e50]
  validation = []
  for n in norm:
    for inverse_lambda in inverse_lambdas:
      lr = LogisticRegression()
      if norm == 1:
        lr = LogisticRegression(penalty = "l1", C = inverse_lambda)
      else:
        lr = LogisticRegression(penalty = "l2", C = inverse_lambda)
      lr.fit(data["Xtrain"], data["Ytrain"])
      validation_score = lr.score(data["Xvalidate"], data["Yvalidate"])
      test_score = lr.score(data["Xtest"], data["Ytest"])
      validation.append((n, inverse_lambda, validation_score, test_score))
  validation.sort(key = lambda x: x[2], reverse = True)
  return validation

def linearSVMModelSelection(data):
  results = []
  for C in [0.001, 0.01, 1, 10, 100]:
    # print "We are working with " + str(C)
    results.append(wrapper_linear(data, C))
  return results

print linearSVMModelSelection(data_1vs7)



