"""
Module contains defintions for a general feed forward neural network
which is trained using stochastic gradient descent, and whose learning
gradients is found using backpropagation which utilizes a quadratic cost
function and sigmoid activation function for simplicity
"""

import random
import numpy as np

class Network(object):
    """
    General Network class that defines a fully connected feed 
    forward NN (aka MultiLayerPerceptron)

    Usage: net = Network(sizes = [2,3,1]) creates a 3 layer NN consisting
    of a 2 neuron input layer, a 3 neuron hidden layer and a 1 neuron output
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return the output vector of network if "a" is given as an input.
        Think about it, a is the "activation" of the first layer as its
        the input. 

        Params:
        a: array-like denoting the input into the neural network.
            Needs to be the size of the first (input) layer of the Network
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return(a)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """
        Train the NN using stochastic gradient descent (ie. minibatch)
    
        Params
        ------
        training_data: list of tuples (x,y) representing the training input and desired output vectors.
                    The other non-optional parameters are pretty self explanatory
        eta: the learning rate
        test_data: a list of tuples (x,y) representing the test inputs and the outputs.
                If provided then the network will be evalutated against the test data after 
                each epoch of training.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            # initialize the random mini batches
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
        Updates the networks weights and biases by finding the gradient of the cost function with respect to
        the weights and biases. Critically we have to consider that the gradients are computed for each training
        example in the mini-batch, then averaged over the mini-batch to tune the weights for that mini-batch. 
        This is built on the assumptions that the cost function can be written as an average of the cost functions
        of individual training examples.

        Params:
        -------

        mini-batch: is a list of tuples (x,y) of the mini batch
        eta: the learning rate
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        for x,y in mini_batch:
            # find the gradient of the cost function with respect to the weights 
            # and bias via backprop
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) 
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # update weight/bias update found going
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # through the individual example gradients

        # Now update the weights and biases of the network
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """
        Returns a tuple (nabla_b, nabla_w) representing the gradient of the cost function C_x
        ie for a single training input x and its corresponding label y

        nabla_b and nabla_w are lists containing numpy arrays arranged as the layers in the network,
        similar to self.biases and self.weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        ################
        # Forward Pass #
        ################
        activation = x
        activations = [x] # list to store activations layer by layer
        zs = [] # list to store weighted input

        # compute the weighted inputs and activations for each layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #################
        # Backward Pass #
        #################
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # BP 1
        nabla_b[-1] = delta                                                      # BP 3
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())                 # BP 4

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # BP 2
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        """
        Returns the number of test inputs for which the neural network outputs the correct result. 
        For newbs, note that the neural network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation. 
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for x,y in test_data] # note how np.argmax returns the index of the highest value in the list
        return sum(int(x==y) for x,y in test_results)


    def cost_derivative(self, output_activations, y):
        """
        Returns the vector of partial derivatives dC_x / da for the output activations
        """
        return (output_activations-y)

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