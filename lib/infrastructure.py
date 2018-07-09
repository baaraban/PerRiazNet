import numpy as np


class Connection:
    def __init__(self, layer, in_size, out_size):
        self._layer = layer
        self._in_size = in_size
        self._out_size = out_size
        self._weights = np.random.rand(out_size, in_size)
        self._biases = np.random.rand(out_size, 1)

    def get_input(self):
        first = self._weights.dot(self._layer.get_result())
        return first + self._biases

    def update_weights(self, deltas, learning_rate):
        old_weights = self._weights.copy()
        self._weights -= learning_rate * deltas.dot(self._layer.get_result().T)
        self._layer.backpropagate(old_weights.T.dot(deltas), learning_rate)

    def update_biases(self, deltas, learning_rate):
        self._biases -= learning_rate * np.mean(deltas, axis=1, keepdims=True)


class Layer:
    def __init__(self, activation_func, out_size):
        self._func = activation_func
        self.out_size = out_size
        self._results = np.zeros(out_size)
        self._last_input = []
        self._connections = []

    def get_result(self):
        return self._results

    def add_connection(self, con):
        self._connections.append(con)

    def forward(self):
        to_pass_in = self._connections[0].get_input()
        for i in range(1, len(self._connections)):
            to_pass_in += self._connections[i].get_input()
        self._last_input = to_pass_in
        self._results = self._func.calc(to_pass_in)

    def backpropagate(self, previous, learning_rate):
        delta = self._func.calc_derivative(self._last_input) * previous
        for con in self._connections:
            con.update_weights(delta, learning_rate)
            con.update_biases(delta, learning_rate)


class InputLayer(Layer):
    def __init__(self, input_size):
        self._input_size = input_size
        self._results = []

    def forward(self, x):
        self._results = x.T

    def get_result(self):
        return self._results

    def backpropagate(self, error, learning_rate):
        return


class OutputLayer(Layer):
    def __init__(self, activation_func, out_size, error_func):
        self._func = activation_func
        self._error_func = error_func
        self.out_size = out_size
        self._results = np.zeros(out_size)
        self._last_input = []
        self._connections = []

    def backpropagate(self, target, learning_rate):
        # currently hardcoded for softmax with cross entropy loss function
        # TODO: Make it generic
        delta = self._results.copy()
        delta -= target
        for con in self._connections:
            con.update_weights(delta, learning_rate)
            con.update_biases(delta, learning_rate)

    def get_error(self, target):
        return self._error_func.calc(target, self._results)


class NeuralNet:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, x):
        self._layers[0].forward(x)
        for i in range(1, len(self._layers)):
            self._layers[i].forward()
        return self._layers[-1].get_result()

    def train(self, train_data, train_results, num_epochs, learning_rate, minibatch_size=1, log_in_console = True):
        for i in range(num_epochs):
            j = 0
            while j + minibatch_size < len(train_data):
                batch = train_data[j:j+minibatch_size]
                self.feed_forward(batch)
                self.backpropagate(train_results[j:j+minibatch_size].T, learning_rate)
                j += minibatch_size
            batch = train_data[j:]
            self.feed_forward(batch)
            self.backpropagate(train_results[j:].T, learning_rate)

            self.feed_forward(train_data)
            error = self._layers[-1].get_error(train_results.T)
            if log_in_console:
                print("Epoch: {0}; Error: {1};".format(i, error))

    def predict(self, x):
        return self.feed_forward(self, x)

    def backpropagate(self, target, learning_rate):
        self._layers[-1].backpropagate(target, learning_rate)
