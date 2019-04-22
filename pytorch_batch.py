"""
This module contains the definition of a simple neural network for pedagogical purpose in PyTorch.
It is supposed to be identical to the simpleNN network class. Because of PyTorch's shortcuts to
computing gradients using autograd, there is no explicit backwards function like in our other module.

Author: Paul Scherer (2019)
Date: April 2019
"""
import random
import numpy as np

# Pytorch tensors and neural network packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# pytorch optimizers
import torch.optim as optim

# Data Loader
import mnist_loader_python3
training_data, _, test_data = mnist_loader_python3.load_data_wrapper()

# Have to change our data set into Pytorch tensors to work.
# torch_training_data = []
# for i in range(len(training_data)):
# 	x, y = training_data[i]
# 	torch_training_data.append((torch.from_numpy(x),(torch.from_numpy(y)).view(10)))

torch_test_data = []
for i in range(len(test_data)):
	x, y = test_data[i]
	torch_test_data.append((torch.from_numpy(x), torch.as_tensor(y)))

def create_mini_batches(training_data, batch_size):
	"""
	Small helper function to create mini-batched training data set
	for mini-batch stochastic gradient descent
	"""
	batches = [training_data[k:k+batch_size] for k in range(0,len(training_data),batch_size)]
	torch_compliant_batches = [] # torch likes batches to be in the shape tensor[num, (dimsIN)]
	for mini_batch in batches:
		mini_batch_obs = []
		mini_batch_labels = []
		for obs in mini_batch:
			x,y = obs
			mini_batch_obs.append(x.flatten())
			mini_batch_labels.append(y.flatten())
		torch_compliant_batches.append((torch.from_numpy(np.array(mini_batch_obs)), torch.from_numpy(np.array(mini_batch_labels))))

	return torch_compliant_batches

# Architecture definition
class SimpleNet(nn.Module):
	def __init__(self):
		super(SimpleNet, self).__init__()
		self.fc1 = nn.Linear(784, 64)
		self.fc2 = nn.Linear(64, 10)

	def forward(self,x):
		x = F.sigmoid(self.fc1(x))
		x = self.fc2(x)
		return x

# Define the Loss and Optimizer
net = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

##################################
###### Training the Network ######
##################################
for epoch in range(100):
	running_loss = 0.0
	random.shuffle(training_data)
	torch_training_data = create_mini_batches(training_data, 1)
	for x_batch, y_batch in torch_training_data:
		optimizer.zero_grad()

		# forward pass
		outputs = net(x_batch)

		# backward pass
		loss = criterion(outputs, y_batch)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
	if epoch % 2 == 0:
		print('[%d] Loss: %.3f' % (epoch, running_loss))

print ("\n #### Finished Training ### \n")

########################
###### Evaluation ######
########################
correct = 0
total = 0
with torch.no_grad():
	for x,y in torch_test_data:
		x = x.view(784) # single input
		outputs = net(x)
		maxvalue, predicted_index = torch.max(outputs.data, 0)
		total += 1
		if predicted_index == y:
			correct += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
