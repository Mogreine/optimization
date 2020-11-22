import numpy as np
from scipy.special import expit
from src.hw2.oracle import make_oracle
from src.hw2.optimization import gradient_descent, newton, newton_hess_free
from src.hw2.oracle import Oracle
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import timeit
import plotly.graph_objs as go
import plotly


def get_w_true(path):
    X, y = load_svmlight_file(path)
    ones_col = np.ones((X.shape[0], 1))
    X = hstack((X, ones_col))

    # initially y = [-1, +1]^n, we map it into y = [0, 1]^n
    y = (y + 1) // 2

    clf = LogisticRegression(penalty='none',
                             tol=1e-8,
                             max_iter=10000,
                             random_state=1,
                             fit_intercept=False
                             )
    clf.fit(X, y)
    w_sk = clf.coef_[0]
    return w_sk


def _plot(data, graph_name):
    names_map = {
        'time': 0,
        'calls': 1,
        'iters': 2
    }
    ind = names_map[graph_name]
    traces = []
    colors = {
        'gs': {'color': 'rgba(255, 0, 0, 0.8)'},
        'brent_scipy': {'color': 'rgba(0, 255, 0, 0.8)'},
        'armijo': {'color': 'rgba(0, 0, 255, 0.8)'},
        'wolfe': {'color': 'rgba(255, 0, 255, 0.8)'},
        'lip': {'color': 'rgba(0, 0, 0, 0.8)'}
    }
    for method_name, method_data in data.items():
        traces.append(
            go.Scatter(
                x=method_data[ind + 2],
                y=method_data[1],
                mode='lines+markers',
                marker=colors[method_name],
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


def run_tests(oracle, w_opt, optimizer='newton', line_search_methods=None, max_iter=10000, tol=1e-16):
    line_search_methods = ['gs', 'brent_scipy', 'armijo', 'wolfe', 'lip'] if line_search_methods is None \
                          else line_search_methods
    if optimizer != 'gd' and 'lip' in line_search_methods:
        line_search_methods.remove('lip')
    optimizers = {
        'gd': gradient_descent,
        'newton': newton,
        'hfn': newton_hess_free
    }

    res = {}
    for method in line_search_methods:
        w_0 = np.zeros(oracle.features)
        # w_0 = np.random.normal(0, 1, oracle.features)
        # w_0 = np.random.uniform(-1 / np.sqrt(oracle.features), 1 / np.sqrt(oracle.features), size=oracle.features)
        # w_0 = np.ones(oracle.features)
        w_pred, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr = optimizers[optimizer](oracle,
                                                                                              w_0,
                                                                                              line_search_method=method,
                                                                                              max_iter=max_iter,
                                                                                              tol=tol)
        print(f'{method}: {iters_arr[-1]}, {oracle.value(w_pred)}')
        rk_arr = np.log10(np.abs(rk_arr - oracle.value(w_opt)))
        res[method] = (w_pred, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr)
    return res


def test_optimization(path, dataset_name=None):
    w_opt = get_w_true(path)
    oracle = make_oracle(path, dataset_name=dataset_name)
    res = run_tests(oracle, w_opt, optimizer='gd', line_search_methods=None, max_iter=10000, tol=1e-8)
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


path = 'hw2/data/cancer.txt'

# test_shit()
test_optimization(path, 'cancer')
# bench()
