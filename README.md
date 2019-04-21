# simpleNN

Simple Neural Network Implementation using basic Python and Numpy for Pedagogic Purposes. 

This repository contains implementations of very simple neural networks inspired by Michael Nielsen's 2015 *Neural Networks and Deep Learning*. It's primary purpose is to highlight the construction of the forward pass and the backward pass through the layers in the implementation. As a bonus the same neural network is also implemented using PyTorch and TensorFlow, and Keras (with TensorFlow) to allow comparision.

## Pre-requisites
Python, all versions 2.5x+ should work. Most of the code will be written for 2.7.12 however care has been put in so that it should run in 3.2+ with little to no alteration.

## Motivation
Reviewing Key Concepts and Preparing materials to teach in Class.

## Currently included:
- Vanilla Feed Forward MLP (with stochastic gradient descent, sigmoid activation layers, quadratic cost function) found in `simpleNN`
- Vanilla Feed Forward MLP (with stochastic gradient descent, sigmoid activation layers, cross entropy cost function, softmax output, L2 regularization) found in `simpleNN2`
- Vanilla Feed Forward MLP (with stochasting gradient descent, sigmoid activation layers, cross entropy cost function, softmax output, L2 regularization) found in `simpleNN2_matrix`

### Bonus
- PYTORCH Vanilla Feed Forward MLP (with stochastic gradient descent, sigmoid activation layers, quadratic cost function) found in `pytorch_nn` (for online learning)
	- `pytorch_batch_nn` is identical to pytorch_nn, but includes batch learning for batch stochastic gradient descent 

#### TODO
- TENSORFLOW Vanilla Feed Forward MLP (with stochastic gradient descent, sigmoid activation layers, quardratic cost function) found in `tensorflow_nn`)
- KERAS (w. Tensorflow) Vanilla Feed Forward MLP (with stochastic gradient descent, sigmoid activation layers, quadratic cost function found in `keras_nn`)