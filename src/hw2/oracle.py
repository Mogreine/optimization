import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import diags


# gotta test it
def make_oracle(path, format='libsvm'):
    if format == 'libsvm':
        X, y = load_svmlight_file(path)
        # initially y = [-1, +1]^n, we map it into y = [0, 1]^n
        y = (y + 1) // 2
        return Oracle(X, y)


class Oracle:
    def __init__(self, X, y):
        self._X = X.copy()
        self._X = self._X.T
        self._y = np.array(y, copy=True)
        # because we look at weights
        self.N = self._X.shape[0]
        assert len(self._y.shape) == 1, "y should be a vector: len(y.shape) is not 1"

    @staticmethod
    def sigmoid(x: np.array):
        return 1 / (1 + np.exp(-x))

    def sigmoid_w(self, x: np.array, w: np.array):
        return self.sigmoid(w.dot(x))

    # returns vector
    def sigmoid_X(self, w: np.array) -> np.array:
        # return np.array([self.sigmoid_w(row, w) for row in self._X])
        return self.sigmoid(self._X.T @ w)

    # returns matrix D: d_ii = sig_w(x) * (1 - sig_w(x))
    def sigmoid_X_diag(self, w: np.array) -> np.ndarray:
        sigs = self.sigmoid_X(w)
        sigs = sigs * (1 - sigs)
        # return np.diagflat([sig * (1 - sig) for sig in sigs])
        return diags(sigs)

    def value(self, w):
        assert len(w.shape) == 1, "x should be a vector: len(x.shape) is not 1"
        sig_x = self.sigmoid_X(w)

        # sum of y_i*log(sig_w(x_i))
        ylog = np.log(sig_x) @ self._y

        # sum of (1 - y_i)*log(1 - sig_w(x_i))
        ylog2 = (1 - self._y) @ np.log(1 - sig_x)

        return -(ylog + ylog2) / self.N

    def grad(self, w):
        return self._X @ (self.sigmoid_X(w) - self._y) / self.N

    def hessian(self, w):
        return self._X.dot(self.sigmoid_X_diag(w)).dot(self._X.T) / self.N

    def hessian_vec_product(self, x, d):
        pass

    def fuse_value_grad(self, w):
        return self.value(w), self.grad(w)

    def fuse_value_grad_hessian(self, w):
        return self.value(w), self.grad(w), self.hessian(w)

    def fuse_value_grad_hessian_vec_product(self, w, d):
        return self.value(w), self.grad(w), self.hessian_vec_product(w, d)


class OracleTester:
    def __init__(self, oracle):
        self.oracle = oracle

    # kinda symmetric grad test
    def test_grad(self, w, d, eps=1e-8):
        f_d = self.oracle.value(w + eps * d)
        f_w = self.oracle.value(w - eps * d)

        fwd_grad_true = (f_d - f_w) / (2 * eps) + eps / 2
        fwd_grad_test = self.oracle.grad(w) @ d

        error = np.linalg.norm(fwd_grad_test - fwd_grad_true)

        return error

    # kinda symmetric hessian test
    def test_hessian(self, w, d, eps=1e-8):
        f_d = self.oracle.grad(w + eps * d)
        f_w = self.oracle.grad(w - eps * d)

        fwd_hessian_true = (f_d - f_w) / (2 * eps) + eps / 2
        fwd_hessian_test = self.oracle.hessian(w) @ d

        error = np.linalg.norm(fwd_hessian_test - fwd_hessian_true)

        return error
