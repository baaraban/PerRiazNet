import numpy as np

LEARNING_RATE = 0.02


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

    def update_weights(self, deltas):
        old_weights = self._weights.copy()
        self._weights -= LEARNING_RATE * deltas.dot(self._layer.get_result().T)
        return old_weights

    def update_biases(self, deltas):
        self._biases -= LEARNING_RATE * np.mean(deltas, axis=1, keepdims=True)


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

    def backpropagate(self, previous):
        delta = self._func.calc_derivative(self._last_input) * previous
        for con in self._connections:
            old_weights = con.update_weights(delta)
            con.update_biases(delta)
        return old_weights.T.dot(delta)


class InputLayer(Layer):
    def __init__(self, input_size):
        self._input_size = input_size
        self._results = []

    def forward(self, x):
        self._results = x.T

    def get_result(self):
        return self._results

    def backpropagate(self, error):
        return


class OutputLayer(Layer):
    def __init__(self, activation_func, out_size, error_func):
        self._func = activation_func
        self._error_func = error_func
        self.out_size = out_size
        self._results = np.zeros(out_size)
        self._last_input = []
        self._connections = []

    def backpropagate(self, target):
        delta = self._results - target
        for con in self._connections:
            old_weights = con.update_weights(delta)
            con.update_biases(delta)
        return old_weights.T.dot(delta)

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

    def train(self, train_data, train_results, num_epochs, minibatch_size=1):
        for i in range(num_epochs):
            batch_indices = np.random.randint(low=0, high=len(train_data), size=(minibatch_size,))
            batch = train_data[batch_indices]
            self.feed_forward(batch)
            error = self._layers[-1].get_error(train_results[batch_indices].T)
            self.backpropagate(train_results[batch_indices].T)
            print("Epoch: " + str(i) + "; Error: " + str(error))

    def predict(self, x):
        return self.feed_forward(self, x)

    def backpropagate(self, target):
        to_pass = target
        for x in reversed(self._layers):
            to_pass = x.backpropagate(to_pass)
