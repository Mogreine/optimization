import numpy as np
import scipy
from sklearn.datasets import load_svmlight_file
from pandas import read_csv
from scipy.sparse import diags
from scipy.sparse import hstack
from scipy.special import expit
from scipy.sparse.linalg import svds, norm


def make_oracle(dataset_path, dataset_name=None, format='libsvm'):
    if format == 'libsvm':
        X, y = load_svmlight_file(dataset_path)
        ones_col = np.ones((X.shape[0], 1))
        X = hstack((X, ones_col))

        if dataset_name == 'a1a':
            # initially y = [-1, +1]^n, we map it into y = [0, 1]^n
            y = (y + 1) // 2
        if dataset_name == 'cancer':
            # initially y = [2, 4]^n, we map it into y = [0, 1]^n
            y = y // 2 - 1

        return Oracle(X, y)
    else:
        data = read_csv(dataset_path, delimiter='\t')
        data_np = data.to_numpy()
        X, y = data_np[:, 1:], data_np[:, 0]
        ones_col = np.ones((X.shape[0], 1))
        X = hstack((X, ones_col))
        return Oracle(X, y)


class Oracle:
    def __init__(self, X, y):
        self._X = X.copy()
        self._y = y.copy()

        self.features = self._X.shape[1]
        self.samples = self._X.shape[0]
        assert len(self._y.shape) == 1, "y should be a vector: len(y.shape) is not 1"

    @staticmethod
    def sigmoid(x: np.array):
        return expit(x)

    # returns vector
    def sigmoid_X(self, w: np.array) -> np.array:
        # return np.array([self.sigmoid_w(row, w) for row in self._X])
        return self.sigmoid(self._X @ w)

    # returns matrix D: d_ii = sig_w(x) * (1 - sig_w(x))
    def sigmoid_X_diag(self, w: np.array) -> np.ndarray:
        sigs = self.sigmoid_X(w)
        sigs = sigs * (1 - sigs)
        return np.diagflat(sigs)

    def value(self, w):
        assert len(w.shape) == 1, "x should be a vector: len(x.shape) is not 1"
        sig_x = self.sigmoid_X(w)

        # sum of y_i*log(sig_w(x_i))
        ylog = np.log(sig_x + 1e-20) @ self._y

        # sum of (1 - y_i)*log(1 - sig_w(x_i))
        ylog2 = (1 - self._y) @ np.log(1 - sig_x + 1e-20)

        return -(ylog + ylog2) / self.samples

    def grad(self, w):
        return 1 / self.samples * self._X.T @ (self.sigmoid_X(w) - self._y)

    def hessian(self, w):
        return 1 / self.samples * self._X.T @ self.sigmoid_X_diag(w) @ self._X

    def hessian_vec_product(self, w, d):
        eps = 1e-5
        f_d = self.grad(w + eps * d)
        f_w = self.grad(w - eps * d)

        fwd_hessian_true = (f_d - f_w) / (2 * eps)
        return fwd_hessian_true

    def fuse_value_grad(self, w):
        return self.value(w), self.grad(w)

    def fuse_value_grad_hessian(self, w):
        return self.value(w), self.grad(w), self.hessian(w)

    def fuse_value_grad_hessian_vec_product(self, w, d):
        return self.value(w), self.grad(w), self.hessian_vec_product(w, d)

    def fuse_grad_hessian(self, w):
        return self.grad(w), self.hessian(w)

    def fuse_grad_hessian_vec_product(self, w, d):
        return self.grad(w), self.hessian_vec_product(w, d)
