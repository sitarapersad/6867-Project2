# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 00:52:43 2016

@author: Sitara
"""

#PROBLEM 3: Support Vector Machine with Pegasos

'''
The following pseudo-code is a slight variation on the Pegasos learning 
algorithm, with a ﬁxed iteration count and non-random presentation of the 
training data. Implement it, and then add a bias term (w0) to the hypothesis, 
but take care not to penalize the magnitude of w0. Your function should output
 classiﬁer weights for a linear decision boundary.
 '''
 
 def pegasosSVM(data, L, max_epochs):
     t = 0
     # Initialise w
     w = 