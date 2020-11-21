import numpy as np
from scipy.special import expit
from src.hw2.oracle import make_oracle
from src.hw2.optimization import gradient_descent, newton, newton_hess_free
from src.hw2.oracle import Oracle
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import timeit
import numpy as np
import plotly.graph_objs as go
import plotly


a = np.array([-1000000000000, 2, 3])
b = np.array([1, 1, 1])


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
        w_opt, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr = method_data

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


def run_tests(oracle, w_opt, line_search_methods=None, max_iter=10000, n_tests=1):
    line_search_methods = ['gs', 'brent_scipy', 'armijo', 'wolfe'] if line_search_methods is None else line_search_methods
    time = []
    rk = []
    res = {}
    for method in line_search_methods:
        w_0 = np.random.normal(0, 1, oracle.features)
        # w_0 = np.zeros(oracle.features)
        # w_0 = np.random.uniform(-1 / np.sqrt(oracle.features), 1 / np.sqrt(oracle.features), size=oracle.features)
        # w_0 = np.ones(oracle.features)
        w_opt, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr = gradient_descent(oracle,
                                                                                        w_0,
                                                                                        line_search_method=method,
                                                                                        max_iter=max_iter,
                                                                                        tol=1e-8)
        print(f'{method}: {iters_arr[-1]}, {oracle.value(w_opt)}')
        res[method] = (w_opt, np.log10(np.abs(rk_arr - oracle.value(w_opt))), elapsed_time_arr, oracle_calls_arr, iters_arr)
    return res


def get_w_true():
    X, y = load_svmlight_file('hw2/data/a1a.txt')
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


def test_optimization():
    w_opt = get_w_true()
    oracle = make_oracle('hw2/data/a1a.txt')
    res = run_tests(oracle, w_opt, line_search_methods=None, max_iter=10000)
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


def bench():
    x = np.random.normal(0, 1, 1000)
    norm = lambda x: np.sqrt(x @ x)
    print(timeit.timeit('import numpy as np; x = np.random.normal(0, 1, 100000); norm = lambda x: np.sqrt(x @ x); norm(x)', number=1000))
    print(timeit.timeit('import numpy as np; x = np.random.normal(0, 1, 100000); norm = np.linalg.norm; norm(x)', number=1000))



# test_shit()
test_optimization()
# bench()
