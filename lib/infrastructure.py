import numpy as np


class Connection:
    def __init__(self, layer, in_size, out_size):
        self._layer = layer
        self._in_size = in_size
        self._out_size = out_size
        self._weights = np.random.rand(out_size, in_size)
        self._biases = np.random.rand(1, out_size)

    def get_input(self):
        return np.dot(self._weights, self._layer.get_result()) + self._biases

    def update_weights(self, deltas):
        return

    def update_biases(self, deltas):
        return


class InputLayer:
    def __init__(self, input_size):
        self._input_size = input_size
        self._results = []

    def forward(self, x):
        self._results = x

    def get_result(self):
        return self._results


class Layer:
    def __init__(self, activation_func, out_size):
        self._func = activation_func
        self.out_size = out_size
        self._results = np.zeros(out_size)
        self._connections = []

    def get_result(self):
        return self._results

    def add_connection(self, con):
        self._connections.append(con)

    def forward(self):
        to_pass_in = np.zeros(self.out_size)
        for con in self._connections:
            to_pass_in += con.get_input()
        self._results = self._func.calc(to_pass_in)

    def backpropagate(self):
        return


class NeuralNet:
    def __init__(self, in_size, out_size, error_func):
        self.error_func = error_func
        self.in_size = in_size
        self.out_size = out_size
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, x):
        self._layers[0].forward(x)
        for i in range(1, len(self._layers)):
            self._layers.forward()
        return self._layers[-1].get_result()

    def train(self, train_data, train_results, num_epochs, minibatch_size = 1):
        for i in range(num_epochs):
            i = np.random.randint(train_data.shape[0])
            train_sample = [train_data[i]]
            self.backpropagate(train_results, self.feed_forward())
            self.update_weights()

    def predict(self, x):
        return self.feed_forward(self, x)

    def backpropagate(self, target, predicted):
        error = self.error_func.calc(target, predicted)







