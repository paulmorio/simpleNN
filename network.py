# General Network Class

import numpy as np

class Network(object):
	"""
	General Network Class which defines a fully connected feed forward NN (saka: MLP)

	Usage: net = Network(sizes = [2,3,1]) creates a 3 layer nn consisting of a 2 neuron
			input layer
	"""

	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]


	def feedforward(self, a):
		"""Return the output of the network if "a" is given as input."""
		for b, w in zip (self.biases, self.weights):
			a = sigmoid(np.dot(w,a)+b)
		return(a)


def sigmoid(z):
	return (1.0/(1.0+np.exp(-z)))

