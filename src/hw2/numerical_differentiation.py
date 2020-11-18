import numpy as np
import scipy
from src.hw2.oracle import make_oracle, OracleTester


a = np.array([1, 2, 3])
b = np.array([1, 1, 1])

X = np.array([*range(9)]).reshape((3, 3))
# print(np.diagflat(a))

oracle = make_oracle('data/a1a.txt')
tester = OracleTester(oracle)


def test_derivative(f):
    errors = []
    for i in range(50):
        w = np.random.normal(0, 1, oracle.features)
        w /= np.linalg.norm(w)

        d = np.random.normal(0, 1, oracle.features)
        d /= np.linalg.norm(d)

        err = f(w, d)
        errors.append(err)
    print(f'Mean error: {np.max(errors)}')


test_derivative(tester.test_grad)
test_derivative(tester.test_hessian)



