import numpy as np


class ActivationFunction:
    name = "base"

    def calc(self, x): return 0

    def calc_derivative(self, x): return 0


class SoftMax(ActivationFunction):
    name = "SoftMax"

    def calc(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def calc_derivative(self, x):
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)


class HyperbolicTangent(ActivationFunction):
    name = "Hyperbolic Tangent"

    def calc(self, x):
        return np.tanh(x)

    def calc_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2


class ReLU(ActivationFunction):
    name = "ReLU"

    def calc(self, x):
        return x * (x > 0)

    def calc_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class ErrorFunction:
    name = "base_error"

    def calc(self, target, prediction):
        return 0

    def calc_derivative(self, target, prediction):
        return 0


class SquareRootError(ErrorFunction):
    name = "Root square mean error"

    def calc(self, target, prediction):
        return np.sqrt(((prediction - target) ** 2).mean())

    def calc_derivative(self, target, prediction):
        return


class CrossEntropy(ErrorFunction):
    name = "Cross entropy loss function"

    def calc(self, target, prediction):
        m = target.shape[0]
        p = SoftMax.calc(target)
        log_likelihood = -np.log(p[range(m), target])
        return np.sum(log_likelihood) / m

    def calc_derivative(self, target, prediction):
        m = prediction.shape[0]
        grad = SoftMax.calc(prediction)
        grad[range(m), target] -= 1
        grad = grad / m
        return grad
