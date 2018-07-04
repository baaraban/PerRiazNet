import numpy as np


class Connection:
    def __init__(self, layer, in_size, out_size):
        self._layer = layer
        self._in_size = in_size
        self._out_size = out_size
        self._weights = np.random.rand(out_size, in_size)

    def get_input(self):
        return np.dot(self._weights, self._layer.get_result())


class Layer:
    def __init__(self, activation_func, in_size, out_size):
        self._func = activation_func
        self.in_size = in_size
        self.out_size = out_size
        self._results = np.zeros(out_size)
        self._connections = []

    def add_connection(self, con):
        self._connections.append(con)

    def get_map(self):
        return self._results

    def forward(self):
        to_pass_in = np.zeros(self.out_size)
        for con in self._connections:
            to_pass_in += con.get_map()
        self._results = self._func.calc(to_pass_in)


class NeuralNet:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)


