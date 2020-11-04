import numpy as np
from scipy.special import expit
from src.hw2.oracle import make_oracle, OracleTester
from src.hw2.optimization import gradient_descent, newton
from src.hw2.oracle import Oracle
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack


a = np.array([-1000000000000, 2, 3])
b = np.array([1, 1, 1])


def test_optimization():
    oracle = make_oracle('hw2/data/a1a.txt')

    max_iter = int(1e4)

    its = []
    for i in range(5):
        w_0 = np.random.normal(0, 1, oracle.features)
        w_opt, iters = gradient_descent(oracle, w_0, line_search_method='armijo', max_iter=max_iter, tol=1e-8)
        its.append(iters)

    print(f'Av. iters: {np.mean(its)}')

    X, y = load_svmlight_file('hw2/data/a1a.txt')
    ones_col = np.ones((X.shape[0], 1))
    X = hstack((X, ones_col))

    # initially y = [-1, +1]^n, we map it into y = [0, 1]^n
    y = (y + 1) // 2

    clf = LogisticRegression(penalty='none',
                             tol=1e-8,
                             max_iter=max_iter,
                             random_state=1,
                             fit_intercept=False
                             )
    clf.fit(X, y)
    w_sk = clf.coef_[0]
    # print(w_sk)
    print(f"mine: {oracle.value(w_opt)}")
    print(f"sklearn: {oracle.value(w_sk)}")


def test_shit():
    X = np.array([*range(9)]).reshape((3, 3))
    print(X)
    print(X @ b)


# test_shit()
test_optimization()
