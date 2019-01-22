# General Network Class

import random
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
		"""
		Return the output of the network if "a" is given as input.
		
		Params:
		-------
		a: array-like denoting the input into the neural network. 
			Needs to be the size of the first (input) layer of the NN
		"""
		for b, w in zip (self.biases, self.weights):
			a = sigmoid(np.dot(w,a)+b)
		return(a)

	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
		"""
		Train the NN using mini-batch stochastic gradient descent.

		Params:
		-------
		training_data: list of tuples (x,y) representing the training inputs and the desired outputs. 
						The other non-optional parameteres are self-explanatory
		test_data: a list of tuples (x,y) representing the test inputs and the outputs. 
						If provided then the network will be evaluatied against the
						test data after each epoch of training.
		"""
		if test_data: 
			n_test = len(test_data)
		n = len(training_data)
		
		for j in range(epochs):
			# make the random mini batches
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
		UPdates the the networks's weights and biases by applying gradient descent using backpropagation to a single mini_batch

		Params:
		-------

		mini_batch: is a list of tuples (x,y)
		eta: the learning rate

		"""

		nabla_b  = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for x,y in mini_batch:
			delta_nabla_b,  delta_nabla_w = self.backprop(x,y)
			# sum the changes caused by each example into the change in weights/biases
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		# Now update the weights of the network
		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.weights, nabla_b)]

	def backprop(self, x, y):
		"""
		Returns a tuple (nabla_b, nabla_w) representing the gradient for cost function C_x
		ie for a single training input, atm. 

		nabla_b and nabla_w are layer by layer lists of numpy arrays, similar to the self.biases, 
		and self.weights

		"""
		pass

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
		return (output_actications-y)


# Static Functions
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