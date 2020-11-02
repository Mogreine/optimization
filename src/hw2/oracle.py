import numpy as np
import scipy


def make_oracle(X, y):
    return Oracle(X, y)


class Oracle:
    def __init__(self, X, y):
        self._X = np.array(X, copy=True)
        self._y = np.array(y, copy=True)
        self._w = np.array(self._y.shape[0])
        self._N = self._y.shape[0]
        assert len(self._y.shape) == 1, "y should be a vector: len(y.shape) is not 1"

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_w(self, x: np.array):
        return self.sigmoid(np.dot(self._w, x))

    def sigmoid_X(self):
        return np.array([self.sigmoid_w(row) for row in self._X])

    def value(self, x):
        assert len(x.shape) == 1, "x should be a vector: len(x.shape) is not 1"
        sig_x = self.sigmoid_w(x)

        # sum of y_i*log(sig_w(x_i))
        ylog = self._y * np.log(sig_x)

        # sum of (1 - y_i)*log(1 - sig_w(x_i))
        ylog2 = (1 - self._y) * np.log(1 - sig_x)

        return -(ylog + ylog2) / self._N

    def grad(self, x):
        return x * (self.sigmoid_w(x) - self._y) / self._N

    def hessian(self, x):
        sig_x = self.sigmoid_w(x)
        return np.dot(x, x) * sig_x * (1 - sig_x) / self._N

    def hessian_vec_product(self, x, d):
        pass

    def fuse_value_grad(self, x):
        # return value(x), grad(x)
        pass

    def fuse_value_grad_hessian(self, x):
        # return value(x), grad(x), hessian(x)
        pass

    def fuse_value_grad_hessian_vec_product(self, x, d):
        # return value(x), grad(x), hessian_vec_product(x, d)
        pass
