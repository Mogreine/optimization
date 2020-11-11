import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import eig as scipy_eig
from scipy.sparse.linalg import inv as sparse_inv
from scipy.optimize import line_search as scipy_line_search, brent as scipy_brent
from scipy.sparse.linalg import norm, svds, eigsh


def golden_section(func, a: float, b: float, eps: float = 1e-8):
    l, r = a, b
    K = 0.381966
    x1 = l + K * (r - l)
    x2 = r - K * (r - l)
    f1 = func(x1)
    f2 = func(x2)
    iterations = 0
    while abs(r - l) > eps:
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
    return (r + l) / 2


def brent(oracle, a: float, b: float, eps: float = 1e-3):
    K = 0.381966
    x, w, v = a + K * (b - a), a + K * (b - a), a + K * (b - a)
    d, d2 = 0, 0
    fx = oracle(x)
    fw, fv = fx, fx
    # it = []

    for i in range(50):
        mid = (a + b) / 2
        tol1 = eps * abs(x) + eps / 10
        tol2 = tol1 * 2
        if abs(x - mid) < tol2 - (b - a) / 2:
            break
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
    return x


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


def armijo(f, grad, xk, pk, is_newton=False):
    alpha = 1 if is_newton else 100
    po = 0.8
    c = 0.0001
    xk_grad = grad(xk)

    # armijo condition
    while f(xk + alpha * pk) > f(xk) + c * alpha * xk_grad @ pk:
        alpha *= po

    return alpha, 1


def lip_const(f, grad, xk, pk):
    L = 0.01
    grad_xk = grad(xk)
    while f(xk + 1 / L * pk) > f(xk) + 1 / L * grad_xk @ pk + 1 / 2 / L * pk @ pk:
        L *= 2
    return 2 / L


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


# TODO: сделать такой же формат вывода, как и у scipy_line_search
def line_search(oracle, x_k, p_k=None, is_newton=False, method='wolf', tol=1e-3):
    p_k = p_k if p_k is not None else -oracle.grad(x_k)
    if method == 'brent' or method == 'gs' or method == 'brent_scipy':
        l, r = 0, 1
        f_line = lambda x: oracle.value(x_k + x * p_k)

        if method == 'brent':
            return brent(f_line, l, r, eps=tol), 0
        if method == 'brent_scipy':
            return scipy_brent(f_line, tol=tol), 0
        if method == 'gs':
            return golden_section(f_line, l, r, eps=tol), 0
    if method == 'wolfe':
        return scipy_line_search(oracle.value, oracle.grad, x_k, p_k)
    if method == 'armijo':
        return armijo(oracle.value, oracle.grad, x_k, p_k, is_newton=is_newton)
    if method == 'lip':
        return lip_const(oracle.value, oracle.grad, x_k, p_k), 1


def gradient_descent(oracle, x0, line_search_method='wolf', tol=1e-8, max_iter=int(1e4)):
    iters = 0
    x0_grad = oracle.grad(x0)

    def stop_criterion(x, tol):
        x_grad = oracle.grad(x)
        x_grad_norm = np.linalg.norm(x_grad)
        x0_grad_norm = np.linalg.norm(x0_grad)
        return x_grad_norm ** 2 / x0_grad_norm ** 2 < tol

    x_k = x0.copy()
    while iters < max_iter:
        grad = oracle.grad(x_k)
        p_k = -grad
        # f_line = lambda x: oracle.value(x_k - x * grad)

        alpha = line_search(oracle, x_k, p_k, method=line_search_method, tol=1e-3)[0]
        if alpha is None:
            alpha = 1

        x_k = x_k + alpha * p_k

        if stop_criterion(x_k, tol):
            break
        print(f"{iters}: {oracle.value(x_k)}; a: {alpha}")
        iters += 1
    return x_k, iters


# only for p.d. matrices
def solve_le_cholesky(X, b):
    L = np.linalg.cholesky(X)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.conj().T, y)
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


def solve_conj_hess_free_naive(Ax, b, tol=1e-8):
    xk = np.ones(len(b))
    # xk = np.random.normal(0, 1, len(b))
    rk_free = Ax(xk) - b
    rk = Ax(xk) - b
    pk = -rk
    k = 0
    while np.linalg.norm(rk) > tol:
    # while np.allclose(rk, zero):
        rk_prod = rk @ rk
        Apk = Ax(pk)
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


def newton(oracle, x0, line_search_method='wolf', tol=1e-8, max_iter=int(1e4)):
    iters = 0

    def stop_criterion(x, tol):
        return np.linalg.norm(x) ** 2 < tol

    x_k = x0.copy()
    while iters < max_iter:
        grad = oracle.grad(x_k)

        if stop_criterion(grad, tol):
            break

        hess = oracle.hessian(x_k)
        hess = correct_hessian_addition(hess)
        # hess_inv = np.linalg.inv(hess)
        p_k = -solve_le(hess, grad, method='cholesky')

        alpha = line_search(oracle, x_k, p_k, is_newton=True, method=line_search_method, tol=1e-3)[0]
        if alpha is None:
            alpha = 1

        x_k = x_k + alpha * p_k

        print(f"{iters}: {oracle.value(x_k)}; a: {alpha}")
        iters += 1
    return x_k, iters


def newton_hess_free(oracle, x0, line_search_method='wolf', tol=1e-8, max_iter=int(1e4)):
    iters = 0

    def stop_criterion(x, tol):
        return np.linalg.norm(x) ** 2 < tol

    x_k = x0.copy()
    while iters < max_iter:
        grad = oracle.grad(x_k)

        if stop_criterion(grad, tol):
            break

        hess = oracle.hessian(x_k)
        hess = correct_hessian_addition(hess)
        # хуево, надо сделать подругому
        Hd = lambda d: oracle.hessian_vec_product(x_k, d) + d * 1e-4
        p_k = -solve_conj_hess_free(hess, Hd, grad)

        alpha = line_search(oracle, x_k, p_k, is_newton=True, method=line_search_method, tol=1e-3)[0]
        if alpha is None:
            alpha = 1

        x_k = x_k + alpha * p_k

        print(f"{iters}: {oracle.value(x_k)}; a: {alpha}")
        iters += 1
    return x_k, iters
