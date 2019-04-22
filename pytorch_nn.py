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
torch_training_data = []
torch_test_data = []
for i in range(len(training_data)):
	x, y = training_data[i]
	torch_training_data.append((torch.from_numpy(x),(torch.from_numpy(y)).view(10)))

for i in range(len(test_data)):
	x, y = test_data[i]
	torch_test_data.append((torch.from_numpy(x), torch.as_tensor(y)))


# Architecture definition
class SimpleNet(nn.Module):
	def __init__(self):
		super(SimpleNet, self).__init__()
		self.fc1 = nn.Linear(784, 64)
		self.fc2 = nn.Linear(64, 10)

	def forward(self,x):
		x = x.flatten() # cause at the moment x is [784,1] we want [784]
		x = F.sigmoid(self.fc1(x))
		x = self.fc2(x)
		return x

# Instantiate Network and define the loss and optimizer for this network
net = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

##################################
###### Training the Network ######
##################################
for epoch in range(100):
	running_loss = 0.0
	for x,y in torch_training_data: # essentially online learning
		optimizer.zero_grad()

		# forward pass
		outputs = net(x)

		# backward pass
		loss = criterion(outputs, y)
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
		outputs = net(x)
		maxvalue, predicted_index = torch.max(outputs.data, 0)
		total += 1
		if predicted_index == y:
			correct += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
