import numpy as np
import scipy
from sklearn.datasets import load_svmlight_file
from scipy.sparse import diags
from scipy.sparse import hstack
from scipy.special import expit
from scipy.sparse.linalg import svds, norm


def make_oracle(path, format='libsvm'):
    if format == 'libsvm':
        X, y = load_svmlight_file(path)
        ones_col = np.ones((X.shape[0], 1))
        X = hstack((X, ones_col))

        # initially y = [-1, +1]^n, we map it into y = [0, 1]^n
        y = (y + 1) // 2

        return Oracle(X, y)


class Oracle:
    def __init__(self, X, y):
        self._X = X.copy()
        self._X = self._X.T
        self._y = np.array(y, copy=True)
        self.lip_const = None

        self.features = self._X.shape[0]
        self.samples = self._X.shape[1]
        assert len(self._y.shape) == 1, "y should be a vector: len(y.shape) is not 1"

    def calc_lipschitz(self):
        if self.lip_const is None:
            svd_vals = svds(self._X, k=1, which='LM', return_singular_vectors=True)[1]
            max_svd_val = svd_vals[0]
            self.lip_const = max_svd_val ** 2 / self.samples ** 2

            self.lip_const = norm(self._X @ self._X.T) ** 2 / self.samples ** 2

            self.lip_const = norm(self._X) / 2 / self.samples
        return self.lip_const

    @staticmethod
    def sigmoid(x: np.array):
        return expit(x)

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
        ylog = np.log(sig_x + 1e-12) @ self._y

        # sum of (1 - y_i)*log(1 - sig_w(x_i))
        ylog2 = (1 - self._y) @ np.log(1 - sig_x + 1e-12)

        return -(ylog + ylog2) / self.samples

    def grad(self, w):
        return self._X @ (self.sigmoid_X(w) - self._y) / self.samples

    def hessian(self, w):
        return (self._X @ self.sigmoid_X_diag(w) @ self._X.T / self.samples).toarray()

    def hessian_vec_product(self, w, d):
        eps = 1e-8
        f_d = self.grad(w + eps * d)
        f_w = self.grad(w - eps * d)

        fwd_hessian_true = (f_d - f_w) / (2 * eps)  # + eps / 2
        return fwd_hessian_true

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

        fwd_grad_true = (f_d - f_w) / (2 * eps) # + eps / 2
        fwd_grad_test = self.oracle.grad(w) @ d

        error = np.linalg.norm(fwd_grad_test - fwd_grad_true)
        print(np.allclose(fwd_grad_test, fwd_grad_true))

        return error

    # kinda symmetric hessian test
    def test_hessian(self, w, d, eps=1e-8):
        f_d = self.oracle.grad(w + eps * d)
        f_w = self.oracle.grad(w - eps * d)

        fwd_hessian_true = (f_d - f_w) / (2 * eps) # + eps / 2
        fwd_hessian_test = self.oracle.hessian(w) @ d

        error = np.linalg.norm(fwd_hessian_test - fwd_hessian_true)

        return error
