# Test bed
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network2_matrix_ver
net = network2_matrix_ver.Network([784, 30, 10], cost=network2_matrix_ver.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)