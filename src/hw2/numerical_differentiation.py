import numpy as np
import scipy
from src.hw2.oracle import make_oracle, OracleTester


a = np.array([1, 2, 3])
b = np.array([1, 1, 1])

X = np.array([*range(9)]).reshape((3, 3))
# print(np.diagflat(a))

oracle = make_oracle('data/a1a.txt')
tester = OracleTester(oracle)


def test_func(f, grad, w, d, tol=1e-8) -> bool:
    eps = 1e-5
    f_d = f(w + eps * d)
    f_w = f(w - eps * d)

    fwd_grad_true = (f_d - f_w) / (2 * eps)
    fwd_grad_test = grad(w) @ d

    return np.allclose(fwd_grad_test, fwd_grad_true, atol=tol)


def test_derivative(f, grad):
    is_good = True
    tol = 1e-8
    for i in range(50):
        w = np.random.normal(0, 1, oracle.features)
        w /= np.linalg.norm(w)

        d = np.random.normal(0, 1, oracle.features)
        d /= np.linalg.norm(d)

        err = test_func(f, grad, w, d, tol=tol)
        is_good &= err

    print(f'Passed all tests: {is_good}, tol: {tol}')


test_derivative(oracle.value, oracle.grad)
test_derivative(oracle.grad, oracle.hessian)
