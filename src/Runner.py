import mnist_loader
import network
import time

class run(object):

    def __init__(self, sizes):
        print("Starting...");
        start = time.time()
        training_data, validation_data, test_data = \
	       mnist_loader.load_data_wrapper()
        self.net = network.Network(sizes)
        (self.net).SGD(training_data, 30, 10, 3.0, test_data=test_data)
        end = time.time()
        print("Done!\nTime Taken: " + str(end - start));
