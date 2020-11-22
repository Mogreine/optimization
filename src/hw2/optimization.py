import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import eig as scipy_eig, cho_solve, cho_factor, solve_triangular
from scipy.sparse.linalg import inv as sparse_inv
from scipy.optimize import line_search as scipy_line_search, brent as brent_sc
from scipy.sparse.linalg import norm, svds, eigsh
import time


def golden_section(func, a: float, b: float, eps: float = 1e-8, max_iter=50):
    l, r = a, b
    K = 0.381966
    x1 = l + K * (r - l)
    x2 = r - K * (r - l)
    f1 = func(x1)
    f2 = func(x2)
    iterations = 0
    while abs(r - l) > eps and iterations < max_iter:
        if f1 < f2:
            r = x2
            f2 = f1
            x2 = x1
            x1 = l + K * (r - l)
            f1 = func(x1)
        else:
            l = x1
            f1 = f2
            x1 = x2
            x2 = r - K * (r - l)
            f2 = func(x2)
        iterations += 1
    return (r + l) / 2, iterations + 2


def brent(oracle, a: float, b: float, eps: float = 1e-3):
    K = 0.381966
    x, w, v = a + K * (b - a), a + K * (b - a), a + K * (b - a)
    d, d2 = 0, 0
    fx = oracle(x)
    fw, fv = fx, fx
    oracle_calls = 1

    for i in range(50):
        mid = (a + b) / 2
        tol1 = eps * abs(x) + eps / 10
        tol2 = tol1 * 2
        if abs(x - mid) < tol2 - (b - a) / 2:
            break
        oracle_calls += 1
        p, q, r = 0, 0, 0
        if abs(d2) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            else:
                q = -q
            r = d2
            d2 = d
        if abs(p) < abs(0.5 * q * r) and q * (a - x) < p < q * (b - x):
            # parabolic interpolation
            d = p / q
            u = x + d
            # f should not be evaluated close to a and b
            if u - a < tol2 or b - u < tol2:
                d = tol1 if x < mid else -tol1
        else:
            # golden section step
            d2 = (b if x < mid else a) - x
            d = K * d2
        # f should not be evaluated close to x
        u = x
        if abs(d) > tol1:
            u += d
        elif d > 0:
            u += tol1
        else:
            u -= tol1

        fu = oracle(u)

        if fu < fx:
            # we found a better point
            if u > x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            # the point is not that good but may be it's at least better the the other two points
            if u < x:
                a = u
            else:
                b = u
            if fu < fw or abs(w - x) < eps:
                # if it's the second best
                v, w = w, u
                fv, fw = fw, fu
            elif fu < fv or abs(v - x) < eps or abs(v - w) < eps:
                # if it's the third best
                v = u
                fv = fu
        # it.append(x)
    return x, oracle_calls


# TODO: Доделать метод правильно, чтобы начинал с 1 и шел либо в большую сторону, либо в меньшую
def armijo_vasya(f, grad, xk, pk, is_newton=False):
    alpha = 1
    b1 = 0.9
    b2 = 1.2
    c = 0.0001
    xk_grad = grad(xk)

    # we can increase step
    if f(xk + alpha * pk) < f(xk) + c * alpha * xk_grad @ pk:
        while f(xk + alpha * pk) < f(xk) + c * alpha * xk_grad @ pk:
            alpha *= b2
    else:
        while f(xk + alpha * pk) > f(xk) + c * alpha * xk_grad @ pk:
            alpha *= b1

    return alpha, 1


def brent_scipy(f_line, tol=1e-8):
    res = brent_sc(f_line, tol=tol, full_output=True, maxiter=30)
    return res[0], res[-1]


def armijo(f, grad, xk, pk, is_newton=False):
    alpha = 1 if is_newton else 100
    po = 0.8
    c = 0.0001
    xk_grad = grad(xk)
    xk_val = f(xk)
    oracle_calls = 2

    # armijo condition
    while f(xk + alpha * pk) > xk_val + c * alpha * xk_grad @ pk:
        alpha *= po
        oracle_calls += 1

    return alpha, oracle_calls


def lip_const(f, grad, xk, pk):
    L = 0.01
    xk_grad = grad(xk)
    xk_val = f(xk)
    oracle_calls = 2

    while f(xk + 1 / L * pk) > xk_val + 1 / L * xk_grad @ pk + 1 / 2 / L * pk @ pk:
        L *= 2
        oracle_calls += 1

    return 2 / L, oracle_calls


def line_search(oracle, x_k, p_k=None, is_newton=False, method='wolf', tol=1e-3):
    p_k = p_k if p_k is not None else -oracle.grad(x_k)
    alpha, oracle_calls = 1, 0
    if method == 'brent' or method == 'gs' or method == 'brent_scipy':
        l, r = 0, 1 if is_newton else 100
        f_line = lambda x: oracle.value(x_k + x * p_k)

        if method == 'brent':
            alpha, oracle_calls = brent(f_line, l, r, eps=tol)
        if method == 'brent_scipy':
            alpha, oracle_calls = brent_scipy(f_line, tol=tol)
        if method == 'gs':
            alpha, oracle_calls = golden_section(f_line, l, r, eps=tol)
    if method == 'wolfe':
        alpha, value_calls, grad_calls = scipy_line_search(oracle.value, oracle.grad, x_k, p_k)[:3]
        # look at this shit
        oracle_calls = value_calls
    if method == 'armijo':
        alpha, oracle_calls = armijo(oracle.value, oracle.grad, x_k, p_k, is_newton=is_newton)
    if method == 'lip':
        alpha, oracle_calls = lip_const(oracle.value, oracle.grad, x_k, p_k)
    return alpha, oracle_calls


def is_positive(X):
    try:
        L = np.linalg.cholesky(X)
    except np.linalg.LinAlgError:
        return False
    return True


def is_symmetric(X):
    return np.allclose(X, X.T)


def is_positive_symmetric(X):
    return is_symmetric(X) and is_positive(X)


def correct_hessian_addition(H):
    eps = 1e-4
    I = np.eye(H.shape[0]) * eps
    while not is_positive(H + I):
        I *= 2
    return H + I


# Doesn't work, eigen values do not converge
def correct_hessian_eigvalues(H):
    eps = 1e-3
    eig_val, eig_vec = np.linalg.eigh(H)
    eig_val[eig_val < 0] = eps
    return eig_vec @ np.diagflat(eig_val) @ eig_vec.T


# only for p.d. matrices
def solve_le_cholesky(X, b):
    # L = np.linalg.cholesky(X)
    # y = solve_triangular(L, b)
    # x = solve_triangular(L.conj().T, y)

    L = cho_factor(X, check_finite=False)
    x = cho_solve(L, b, check_finite=False)

    return x


def solve_le_inv(X, b):
    return np.linalg.inv(X) @ b


def solve_conj(A, b, Ax=None, tol=1e-8):
    zero = np.zeros(len(b))
    xk = np.ones(len(b))
    # xk = np.random.normal(0, 1, len(b))
    rk = A @ xk - b
    pk = -rk
    k = 0
    while np.linalg.norm(rk) > tol:
    # while np.allclose(rk, zero):
        rk_prod = rk @ rk
        Apk = A @ pk
        ak = rk_prod / (pk @ Apk)
        xk = xk + ak * pk
        rk = rk + ak * Apk
        bk = rk @ rk / rk_prod
        pk = -rk + bk * pk
        k += 1
    print(f'conj iters: {k}')
    return xk


def solve_le(X, b, method='cholesky'):
    if method == 'cholesky':
        return solve_le_cholesky(X, b)
    if method == 'inv':
        return solve_le_inv(X, b)
    if method == 'conj':
        return solve_conj(X, b)


def gradient_descent(oracle, x0, line_search_method='wolf', tol=1e-8, max_iter=int(1e4)):
    start_time = time.time()
    iters = 0
    x0_grad = oracle.grad(x0)

    oracle_calls_arr = [0]
    elapsed_time_arr = [0]
    iters_arr = [0]
    rk_arr = [oracle.value(x0)]

    def stop_criterion(x, tol):
        x_grad = oracle.grad(x)
        x_grad_norm = np.linalg.norm(x_grad)
        x0_grad_norm = np.linalg.norm(x0_grad)
        rk = x_grad_norm ** 2 / x0_grad_norm ** 2
        return rk < tol

    x_k = x0.copy()
    while iters < max_iter:
        grad = oracle.grad(x_k)
        p_k = -grad

        alpha, oracle_calls_ls = line_search(oracle, x_k, p_k, method=line_search_method, tol=1e-8)
        if alpha is None:
            alpha = 1

        x_k = x_k + alpha * p_k

        elapsed_time_arr.append(time.time() - start_time)
        oracle_calls_arr.append(oracle_calls_arr[-1] + 1 + oracle_calls_ls)
        iters_arr.append(iters_arr[-1] + 1)
        rk_arr.append(oracle.value(x_k))

        if stop_criterion(x_k, tol):
            break
        # print(f"{iters_arr[-1]}: {oracle.value(x_k)}; a: {alpha}")
        iters += 1

    return x_k, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr


def newton(oracle, x0, line_search_method='wolf', tol=1e-8, max_iter=int(1e4)):
    start_time = time.time()
    iters = 0

    x_k = x0.copy()
    oracle_calls_arr = [0]
    elapsed_time_arr = [0]
    iters_arr = [0]
    rk_arr = [oracle.value(x0)]

    def stop_criterion(x, tol):
        return np.linalg.norm(x) ** 2 < tol

    while iters < max_iter:
        grad, hess = oracle.fuse_grad_hessian(x_k)

        if stop_criterion(grad, tol):
            break

        hess = correct_hessian_addition(hess)
        p_k = -solve_le(hess, grad, method='cholesky')

        alpha, oracle_calls_ls = line_search(oracle, x_k, p_k, is_newton=False, method=line_search_method, tol=1e-8)
        if alpha is None:
            alpha = 1

        x_k = x_k + alpha * p_k

        print(f"{iters}: {oracle.value(x_k)}; a: {alpha}")

        elapsed_time_arr.append(time.time() - start_time)
        oracle_calls_arr.append(oracle_calls_arr[-1] + oracle_calls_ls + 1)
        iters_arr.append(iters_arr[-1] + 1)
        rk_arr.append(oracle.value(x_k))

        iters += 1
    return x_k, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr


def newton_hess_free(oracle, x0, line_search_method='wolf', tol=1e-8, max_iter=int(1e4)):
    def solve_conj_hess_free_line_search(Hx, grad, tol=1e-20):
        # в теории должен сойтись не больше чем за n, но гессиан может быть около сингулярным
        norm = lambda x: np.linalg.norm(x)
        max_iter = int(len(grad) * 3)
        eps = min(0.5, np.sqrt(norm(grad))) * norm(grad)
        zk = np.zeros(len(grad))
        # xk = np.random.normal(0, 1, len(b))
        rk = grad
        dk = -rk
        for k in range(max_iter):
            Hdk = Hx(dk)
            dHd = dk @ Hdk

            if dHd < tol:
                if k == 0:
                    zk = -grad
                break

            rk_prod = rk @ rk
            ak = rk_prod / dHd
            zk = zk + ak * dk
            rk = rk + ak * Hdk

            # print(f'{k}: {norm(rk)}')
            if norm(rk) < eps:
                break

            bk = rk @ rk / rk_prod
            dk = -rk + bk * dk
            k += 1
        # print(f'conj iters: {k}')
        return zk, k

    def stop_criterion(x, tol):
        return np.linalg.norm(x) ** 2 < tol

    start_time = time.time()
    iters = 0

    oracle_calls_arr = [0]
    elapsed_time_arr = [0]
    iters_arr = [0]
    rk_arr = [oracle.value(x0)]

    x_k = x0.copy()
    while iters < max_iter:
        grad = oracle.grad(x_k)

        if stop_criterion(grad, tol):
            break

        Hd = lambda d: oracle.hessian_vec_product(x_k, d)
        p_k, hess_vec_prod_calls = solve_conj_hess_free_line_search(Hd, grad)

        alpha, oracle_calls_ls = line_search(oracle, x_k, p_k, is_newton=False, method=line_search_method, tol=1e-8)
        if alpha is None:
            alpha = 1

        x_k = x_k + alpha * p_k

        # print(f"{iters}: {oracle.value(x_k)}; a: {alpha}")

        elapsed_time_arr.append(time.time() - start_time)
        oracle_calls_arr.append(oracle_calls_arr[-1] + oracle_calls_ls + hess_vec_prod_calls + 1)
        iters_arr.append(iters_arr[-1] + 1)
        rk_arr.append(oracle.value(x_k))
        iters += 1
    return x_k, rk_arr, elapsed_time_arr, oracle_calls_arr, iters_arr
