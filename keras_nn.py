"""
Keras implementation of the simpleNN network

Keras is a wrapper over tensorflow (and some other frameworks like 
theano) specially suited for making neural networks.

From tensorflow 2.0 keras like neural network implementation will 
be officially part of the tensorflow package

Author: Paul Scherer
Date: April 2019
"""
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

####################
## Hyperparametes ##
####################
batch_size = 100
output_dim = 10
epochs = 500

######################
## Data Preparation ##
######################
import mnist_loader_python3
training_data, _, test_data = mnist_loader_python3.load_data_wrapper()

# Make data usable with keras, happily we dont have to do any batching 
# ourselves
keras_compliant_train_x = []
keras_compliant_train_y = []
for x, y in training_data:
	keras_compliant_train_x.append(x.flatten())
	keras_compliant_train_y.append(np.argmax(y.flatten()))
x_train = np.array(keras_compliant_train_x)
y_train = np.array(keras_compliant_train_y)

keras_compliant_test_x = []
keras_compliant_test_y = []
for x, y in test_data:
	keras_compliant_test_x.append(x.flatten())
	keras_compliant_test_y.append(y)
x_test = np.array(keras_compliant_test_x)
y_test = np.array(keras_compliant_test_y)

# Change labels into binary categorical vectors
y_train = keras.utils.to_categorical(y_train, output_dim)
y_test = keras.utils.to_categorical(y_test, output_dim)

########################
## Network Definition ##
########################
net = Sequential()
net.add(Dense(64, activation="sigmoid", input_shape=(784,)))
net.add(Dense(10, activation="softmax"))
net.summary()

net.compile(loss="categorical_crossentropy",
			optimizer=SGD(),
			metrics=["accuracy"])

#######################
## Train the Network ##
#######################
history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data = (x_test, y_test))
scores = net.evaluate(x_test, y_test, verbose=0)
print ("Test Loss: %s, Test Accuracy: %s"%(scores[0], scores[1]))