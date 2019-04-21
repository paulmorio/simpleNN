# Import the MNIST Dataset
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Import the new and improved network, instantiate and train!
import simpleNN2_matrix as nn2m
import simpleNN2 as nn2
net = nn2.Network([784, 30, 10], cost=nn2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)