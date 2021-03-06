# Load Data from MNIST
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Import the Network class, instantiate and train!
import simpleNN
net = simpleNN.Network([784, 100, 50, 20, 10])
nabla_b, nabla_w = net.SGD(training_data, 30, 10, 3.0, test_data = test_data)

