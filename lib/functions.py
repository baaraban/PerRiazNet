import numpy as np


class ActivationFunction:
    name = "base"

    @staticmethod
    def calc(x): return 0

    @staticmethod
    def calc_derivative(x): return 0


class SoftMax(ActivationFunction):
    name = "SoftMax"

    @staticmethod
    def calc(x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def calc_derivative(x):
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)


class HyperbolicTangent(ActivationFunction):
    name = "Hyperbolic Tangent"

    @staticmethod
    def calc(x):
        return np.tanh(x)

    @staticmethod
    def calc_derivative(x):
        return 1.0 - np.tanh(x) ** 2


class ReLU(ActivationFunction):
    name = "ReLU"

    @staticmethod
    def calc(x):
        return x * (x > 0)

    @staticmethod
    def calc_derivative(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class Sigmoid(ActivationFunction):
    name = "Sigmoid"

    @staticmethod
    def calc(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def calc_derivative(x):
        return (np.exp(-x)) / ((1 + np.exp(-x)) ** 2)


class ErrorFunction:
    name = "base_error"

    @staticmethod
    def calc(target, prediction):
        return 0

    @staticmethod
    def calc_derivative(target, prediction):
        return 0


class SquareRootError(ErrorFunction):
    name = "Root square mean error"

    @staticmethod
    def calc(target, prediction):
        return np.sqrt(((prediction - target) ** 2).mean())

    @staticmethod
    def calc_derivative(target, prediction):
        return


class CrossEntropy(ErrorFunction):
    name = "Cross entropy loss function"

    @staticmethod
    def calc(target, prediction, epsilon=1e-12):
        m = target.shape[0]
        log_likelihood = -np.sum(np.multiply(target, np.log(prediction)))
        return np.sum(log_likelihood) / m

    @staticmethod
    def calc_derivative(target, prediction):
        return - (target * (1 / prediction) + (1 - target) * (1 / 1 - prediction))