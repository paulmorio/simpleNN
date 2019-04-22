"""
This is a basic pre 2.0 tensorflow version of simpleNN network class

Author: Paul Scherer
Date: April 2019

"""
import tensorflow as tf

##########
## Data ##
##########
import mnist_loader_python3
training_data, _, test_data = mnist_loader_python3.load_data_wrapper()

#####################
## HyperParameters ##
#####################
learning_rate = 0.001
epochs = 500
batch_size = 100

################################
### Network Shape Paramaters ###
################################
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
correct_prediction = tf.equals(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables in their start values
init = tf.global_variables_initializer()


