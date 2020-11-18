import numpy as np
from scipy.special import expit
from src.hw2.oracle import make_oracle, OracleTester
from src.hw2.optimization import gradient_descent, newton, newton_hess_free
from src.hw2.oracle import Oracle
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import timeit


a = np.array([-1000000000000, 2, 3])
b = np.array([1, 1, 1])


def test_optimization():
    oracle = make_oracle('hw2/data/a1a.txt')

    max_iter = int(1e4)

    its = []
    for i in range(5):
        w_0 = np.random.normal(0, 1, oracle.features)
        w_0 = np.zeros(oracle.features)
        w_0 = np.random.uniform(-1 / np.sqrt(oracle.features), 1 / np.sqrt(oracle.features), size=oracle.features)
        # w_0 = np.ones(oracle.features)
        w_opt, iters = newton(oracle, w_0, line_search_method='brent', max_iter=max_iter, tol=1e-8)
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
                             fit_intercept=False,
                             solver='newton-cg'
                             )
    clf.fit(X, y)
    w_sk = clf.coef_[0]
    # print(w_sk)
    print(f"mine   : {oracle.value(w_opt)}")
    print(f"sklearn: {oracle.value(w_sk)}")


def test_shit():
    X = np.array([*range(9)]).reshape((3, 3))
    a = np.array([1, 2, 3])
    print(np.diag(a))
    X = X + X.T
    eig_vals, eig_vecs = np.linalg.eigh(X)
    print(eig_vecs)
    print(X)
    print(eig_vecs @ np.diagflat(eig_vals) @ eig_vecs.T)


def bench():
    x = np.random.normal(0, 1, 1000)
    norm = lambda x: np.sqrt(x @ x)
    print(timeit.timeit('import numpy as np; x = np.random.normal(0, 1, 100000); norm = lambda x: np.sqrt(x @ x); norm(x)', number=1000))
    print(timeit.timeit('import numpy as np; x = np.random.normal(0, 1, 100000); norm = np.linalg.norm; norm(x)', number=1000))



# test_shit()
test_optimization()
# bench()
