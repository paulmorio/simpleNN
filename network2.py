"""
Network 2

adapted from Michael Nielsen 2015
author: paulmorio

Is a more advanced and modular version of the Neural Network Class 
described in network.py

This implements a feedforward neural network which uses stochastic 
gradient descent to learn (the gradients of the weights and biases 
are learnt using backpropagation).

Improvements include the addition of the cross entropy cost function,
regularization techniques, and better initialization of the networks weights.

The original intent of keeping the code simple and educational rather than
optimized for speed applies.
"""

# Library imports
##################
# Standard
import json
import random
import sys

# 3rd Party
import numpy as np

##################
# Classes
##################
class QuadraticCost(object):
    """
    Class which implements the quadratic cost for neural networks
    via static methods
    """

    @staticmethod
    def fn(a,y):
        """Return the cost associated with input 'a' and true label 'y'"""
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z,a,y):
        """Return the error delta from the output layer L"""
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
    """
    Class which implements the cross entropy cost for neural networks
    via static methods
    """
    @staticmethod
    def fn(a,y):
        """Return the cost associated with input 'a' and true label 'y'"""
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z,a,y):
        """Return the error delta from the output layer L"""
        return (a-y)
        

# Static Functions
##################
def sigmoid(z):
    """
    Returns the sigmoid of z
    """
    return (1.0/(1.0+np.exp(-z)))

def sigmoid_prime(z):
    """
    Returns the value of derivative of sigmoid at z
    """
    return sigmoid(z) * (1-sigmoid(z))