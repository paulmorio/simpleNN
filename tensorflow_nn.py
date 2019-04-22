"""
This is a basic pre 2.0 tensorflow version of simpleNN network class

There are many guides and tutorials that say tensorflow creates a computational graph,
and trains the network. which is entirely correct. But lack of additional words leaves 
people in confusion around what the role of python is here in this situation.



Author: Paul Scherer
Date: April 2019

"""
import numpy as np
import random
import tensorflow as tf

##########
## Data ##
##########
import mnist_loader_python3
training_data, _, test_data = mnist_loader_python3.load_data_wrapper()

# Make data usable with Tensorflow
def create_batches(data, batch_size):
	"""
	Returns training data as a list of batches in format [(x_batch, y_batch)]
	where x_batch is the batch of observations and y_batch is the batch of labels
	in one-hot encoding if necessary
	"""
	batches = [training_data[k:k+batch_size] for k in range(0, len(data), batch_size)]
	tf_compliant_batches = []
	for batch in batches:
		mini_batch_obs = []
		mini_batch_labels = []
		for obs in batch:
			x,y = obs
			mini_batch_obs.append(x.flatten())
			mini_batch_labels.append(y.flatten())
		tf_compliant_batches.append((np.array(mini_batch_obs), np.array(mini_batch_labels)))
	return tf_compliant_batches

#####################
## HyperParameters ##
#####################
learning_rate = 0.001
epochs = 500
batch_size = 100

###########################
### Network Definition  ###
###########################
input_dim = 784
hidden_layer_dim = 64
output_dim = 10

# tf graph input
x = tf.placeholder("float", [None, input_dim])
y = tf.placeholder("float", [None, output_dim])

# Weights and biases
weights = {
	"h1": tf.Variable(tf.random_normal([input_dim, hidden_layer_dim])),
	"out": tf.Variable(tf.random_normal([hidden_layer_dim, output_dim]))
}
biases = {
	"b1": tf.Variable(tf.random_normal([hidden_layer_dim])),
	"out": tf.Variable(tf.random_normal([output_dim]))
}

# Define model
def Network(x):
	layer1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
	out_layer = tf.add(tf.matmul(layer1, weights["out"]), biases["out"])
	return out_layer

# Instantiate this model
logits = Network(x)
prediction = tf.nn.softmax(logits)

# Define the loss
loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_fn)

# Evaluation subgraph
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables in their start values
init = tf.global_variables_initializer()

##########################
### Main Control Loop  ###
##########################
with tf.Session() as sess:
	# Run the initializer we created to instantiate the variables
	sess.run(init)

	# The main loop
	for epoch in range(epochs):
		random.shuffle(training_data)
		tf_training_data = create_batches(training_data, batch_size)
		for mini_batch in tf_training_data:
			x_batch, y_batch = mini_batch

			# The full forward and backward pass in one go using the optimizers
			# See that the train_op is the last thing in the computational graph,
			# with x and y being the inputs at the start of that computational graph
			sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

		# Just some display stuff
		if epoch % 10 == 0:
			loss, acc = sess.run([loss_fn, accuracy], feed_dict={x: x_batch, y: y_batch})
			print ("Epoch %s, Loss: %s, Accuracy %s" % (epoch, loss, acc))

	print("## Training Finished ##")

	# #############################
	# ## Evaluation on Test Data ##
	# #############################

	# test_acc = sess.run(accuracy, feed_dict={x: test_x, test_y})
	# print ("Test accuracy: {0}".format(test_acc))