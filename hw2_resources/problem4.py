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
      data["Ytrain"].extend([label*1.0 for i in xrange(200)])
      data["Xvalidate"].extend(digits[200:350,:])
      data["Yvalidate"].extend([label*1.0 for i in xrange(150)])
      data["Xtest"].extend(digits[350:500,:])
      data["Ytest"].extend([label for i in xrange(150)])
  for handle in handles:
    data[handle] = np.array(data[handle])
    if handle[0] == "Y":
      data[handle] = data[handle].reshape(len(data[handle]), 1)
  return data

data_1vs7 = getDataSets([[1], [7]], False)
# for key 
# data_1vs7_normalized = getDataSets([[1], [7]], True)
# data_3vs5 = getDataSets([[3], [5]], False)
# data_3vs5_normalized = getDataSets([[3], [5]], True)
# data_4vs9 = getDataSets([[4], [9]], False)
# data_4vs9_normalized = getDataSets([[4], [9]], True)
# data_evenvsodd = getDataSets([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], False)
# data_evenvsodd_normalized = getDataSets([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], True)

def lrModelSelection(data):
  norm = [2]
  inverse_lambdas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1, 1/5.0, 1/2.0, 1, 2, 5, 10, 1e2, 10e50]
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

print lrModelSelection(data_1vs7)

def linearSVMModelSelection(data):
  results = []
  for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
    results.append(wrapper_linear(data, C))
  results.sort(key = lambda x: x[1], reverse = True)
  return results

# print linearSVMModelSelection(data_1vs7)

def gaussianSVMModelSelection(data):
  results = []
  for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
    for gamma in [2**i for i in range(-2, 3)]:
      results.append(wrapper_gaussian(data, C, gamma))
  results.sort(key = lambda x: x[1], reverse = True)
  return results

print gaussianSVMModelSelection(data_1vs7)

def pegasosGaussianSVMModelSelection(data):
  results = []
  for gamma in [2**i for i in range(-2, 3)]:
    results.append(wrapper(data, gamma))
  results.sort(key = lambda x: x[1], reverse = True)
  return results

# print pegasosGaussianSVMModelSelection(data_1vs7)

