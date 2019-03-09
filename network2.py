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


class Network(object):
    """
    General Network class that defines a fully connected feed 
    forward NN (aka MultiLayerPerceptron). However this iteration allows 
    for several augmentations 

    Usage: net = Network(sizes = [2,3,1]) creates a 3 layer NN consisting
    of a 2 neuron input layer, a 3 neuron hidden layer and a 1 neuron output
    """
     
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        NB: The biases and weights for the network are initialized randomly
        self.default_weight_initializer()
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        Initializes each weight using a Gaussian distribution(0, 1)
        over the square root of the number of the weights connecting
        to the same neuron. This initialization helps the network learn
        faster. The more naive weight initializer is described in large weight initializer

        """
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        Initialise the weights using a Gaussian distribution with mean 0
        and standard deviation of 1. Initializes the biases using a G(0,1)
        """

        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output layer activation of the network given 'a' as input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            lmbda = 0.0, 
            evaluation_data = None,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False):
        """
        Trains the network using mini-batch stochastic gradient descient.
        The lmbda controls the regulaization coefficient described in the
        L2 Regularization

        The method returns a tuple containing four lists: 
        0: the per epoch costs on the evaluation data,
        1: the accuracies on the evaluation data,
        2: the costs on the training data
        3: the accuracies on the training data.

        All values are evaluated at the end of each training epoch.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [],[]
        training_cost, training_accuracy = [],[]
        
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            
            # Mini batch learning
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j

            # Monitoring Tasks
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {}/ {}".format(accuracy, n)

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost of evaluation data: {}".format(cost)

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {}/ {}".format(self.accuracy(evaluation_data), n_data)

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the networks weights and biases by applying gradient descent
        using backpropagation to find gradient vectors of cost wrt weights and biases
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_b, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Returns a tuple (nabla_b, nabla_w) representing 
        the gradient for the cost function C_x
        nabla_b and nabla_w are layer-by-layer lists of 
        numpy arrays similar to self.biases and self.weight
        """

        pass


# Static Functions
##################
def load(filename):
    """
    Loads a neural network from the file, filename. Returns an instance of Network
    """
    pass

def vectorized_result(j):
    """
    Return a 10 dimensional unit vector with a 1.0 
    in the j'th position and zeroes elsewhere. 
    This is used to convert a digit (0...9) into a corresponding
    desired output from the neural network.
    """
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

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