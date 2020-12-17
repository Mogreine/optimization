import numpy as np
from scipy.special import expit
from src.hw2.oracle import make_oracle
from src.hw2.optimization import Optimizer, GradientDescent, Newton, HFN, BFGS, LBFGS, LogRegl1
from src.hw2.optimization import LineSearchOptimizer, GoldenSection, Brent, Armijo, Wolfe, Lipschitz, Wolfe_Arm
from src.hw2.oracle import Oracle
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import timeit
import plotly.graph_objs as go
import plotly
import time

from typing import List


def get_w_true(X, y):
    clf = LogisticRegression(penalty='none',
                             tol=1e-16,
                             max_iter=10000,
                             random_state=1,
                             fit_intercept=False
                             )
    clf.fit(X, y)
    w_sk = clf.coef_[0]
    return w_sk


def _plot(data, graph_name):
    def gen_color():
        return {
            'color': f'rgba({np.random.randint(0, 255)},'
                     f' {np.random.randint(0, 255)},'
                     f' {np.random.randint(0, 255)},'
                     f' 0.8)'
        }

    names_map = {
        'time': 0,
        'calls': 1,
        'iters': 2
    }
    ind = names_map[graph_name]
    traces = []
    colors = {
        'gs': {'color': 'rgba(255, 0, 0, 0.8)'},
        'brent': {'color': 'rgba(0, 255, 0, 0.8)'},
        'armijo': {'color': 'rgba(0, 0, 255, 0.8)'},
        'wolfe': {'color': 'rgba(255, 0, 255, 0.8)'},
        'lipschitz': {'color': 'rgba(0, 0, 0, 0.8)'}
    }
    for method_name, method_data in data.items():
        traces.append(
            go.Scatter(
                x=method_data[ind + 2],
                y=method_data[1],
                mode='lines',
                marker=colors[method_name] if method_name in colors else gen_color(),
                name=method_name
                # xaxis='x1',
                # yaxis='y1'
            )
        )

    layout = go.Layout(
        xaxis=dict(title=graph_name, zeroline=False),
        yaxis=dict(title='$log_{10}|F(w_i) - F(w^*)|$', zeroline=False)
    )

    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.iplot(fig)


def plot(data: dict, graph=('time', 'calls', 'iters')):
    for graph_name in graph:
        _plot(data, graph_name)


def run_tests(oracle, w_opt, optimizer: Optimizer, line_search_methods: List[LineSearchOptimizer],
              line_search_method_names: List, max_iter=10000, tol=1e-16, verbose=0):
    res = {}
    for method, name in zip(line_search_methods, line_search_method_names):
        w_0 = np.zeros(oracle.features)
        # w_0 = np.random.normal(0, 1, oracle.features)
        # w_0 = np.random.uniform(-1 / np.sqrt(oracle.features), 1 / np.sqrt(oracle.features), size=oracle.features)
        w_0 = np.ones(oracle.features) * 5
        w_pred = optimizer.optimize(w_0,
                                    line_search=method,
                                    max_iter=max_iter,
                                    tol=tol,
                                    verbose=verbose)
        rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr = optimizer.get_data()
        print(f'{name}: {iters_arr[-1]}, {oracle.value(w_pred)}')
        rk_arr = np.log10(np.abs(rk_arr - oracle.value(w_opt)))
        res[name] = (w_pred, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr)
    return res


def test_optimization():
    path = 'hw2/data/a1a.txt'
    dataset_name = 'a1a'
    oracle = make_oracle(path, dataset_name=dataset_name, format='libsvm')
    w_opt = get_w_true(oracle._X, oracle._y)
    print(f'sklearn: {oracle.value(w_opt)}')

    optimizer_gd = GradientDescent(oracle)
    optimizer_newton = Newton(oracle, solve='cholesky')
    optimizer_hfn = HFN(oracle)

    line_search_methods = [
        Armijo(po=0.1, c=0.0001),
        Wolfe(c1=0.0001, c2=0.9),
        GoldenSection(tol=1e-8, max_iter=40, bracketing=True),
        Brent(bracketing=True)
    ]
    line_search_names = ['armijo', 'wolfe', 'gs', 'brent']
    res = run_tests(oracle, w_opt, optimizer=optimizer_gd, line_search_methods=line_search_methods,
                    line_search_method_names=line_search_names, max_iter=1000, tol=1e-16, verbose=1)
    plot(res)


def run_tests2(oracle, w_opt, optimizers: List[Optimizer], line_search_methods: List[LineSearchOptimizer],
               line_search_method_names: List, max_iter=10000, tol=1e-16, verbose=0):
    res = {}
    for method_gl, method_ls, name in zip(optimizers, line_search_methods, line_search_method_names):
        w_0 = np.zeros(oracle.features)
        w_0 = np.random.normal(0, 1, oracle.features)
        w_0 = np.random.uniform(-1 / np.sqrt(oracle.features), 1 / np.sqrt(oracle.features), size=oracle.features)
        w_0 = np.ones(oracle.features) * 4
        w_pred = method_gl.optimize(w_0,
                                    line_search=method_ls,
                                    max_iter=max_iter,
                                    tol=tol,
                                    verbose=verbose)
        rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr = method_gl.get_data()
        print(f'{name}: {iters_arr[-1]}, {oracle.value(w_pred)}')
        rk_arr = np.log10(np.abs(rk_arr - oracle.value(w_opt)))
        res[name] = (w_pred, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr)
    return res


def compare_methods():
    path = 'hw2/data/a1a.txt'
    dataset_name = 'a1a'
    oracle = make_oracle(path, dataset_name=dataset_name, format='libsvm')
    w_opt = get_w_true(oracle._X, oracle._y)

    t = time.time()
    print(f'sklearn: {oracle.value(w_opt)}')
    print(f'time: {time.time() - t}')
    global_methods = [
        LogRegl1(oracle, l=0.01),
        GradientDescent(oracle),
        # LBFGS(oracle, 1),
        # BFGS(oracle),
        # Newton(oracle, solve='cholesky'),
        # HFN(oracle),
    ]
    ls_methods = [
        None,
        Lipschitz(),
        # Wolfe(),
        # Wolfe(),
        # Wolfe(),
        # Wolfe()
    ]
    names = [
        'proximal gd \w l1',
        'gd + lipschitz'
        # 'l-bfgs + wolfe',
        # 'bfgs + wolfe',
        # 'newton + wolfe',
        # 'hfn + wolfe',
    ]
    res = run_tests2(oracle, w_opt, optimizers=global_methods, line_search_methods=ls_methods,
                     line_search_method_names=names, max_iter=4000, tol=1e-8, verbose=1)
    plot(res)


def test_shit():
    X = np.array([*range(9)]).reshape((3, 3))
    a = np.array([1, 2, 3])
    print(np.diag(a))
    X = X + X.T
    eig_vals, eig_vecs = np.linalg.eigh(X)
    print(eig_vecs)
    print(X)
    print(eig_vecs @ np.diagflat(eig_vals) @ eig_vecs.T)


def gen_data(n=1000, m=20):
    X = np.random.normal(0, 1, (n, m))
    y = np.random.randint(2, size=n)
    data = np.hstack([y.reshape(-1, 1), X])
    np.savetxt('hw2/data/gen.tsv', X=data, delimiter='\t')


def test():
    a = np.block([
        [1, 2, 3],
        [1, 2] + [0]
    ])
    print(a)


# test()
# test_shit()
# test_optimization()
compare_methods()

# gen_data()
