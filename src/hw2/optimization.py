import numpy as np
from scipy.sparse.linalg import inv as sparse_inv
from scipy.optimize import line_search as scipy_line_search
from scipy.sparse.linalg import norm, svds


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


# TODO: Стоит посмотреть в книге про "c" и мб протестить "po"
def armijo(f, grad, xk, pk, is_newton=False):
    alpha, po = 1 if is_newton else 100, 1 / 2
    c = 0.0001
    xk_grad = grad(xk)

    # armijo condition
    while f(xk + alpha * pk) > f(xk) + c * alpha * xk_grad @ pk:
        alpha *= po

    return alpha, 1


def lip_const(f, grad, xk, pk):
    # return 1 / oracle.calc_lipschitz()
    # return 1 / norm(oracle.hessian(w)) ** 2
    # return 1 / svds(oracle.hessian(w), k=1, which='LM', return_singular_vectors=True)[1][0] ** 2
    # svd_vals = svds(oracle.hessian(w), return_singular_vectors=True)[1]
    # svd_min, svd_max = np.min(svd_vals), np.max(svd_vals)
    # return 2 / (svd_min + svd_max)
    L = 10000
    # while f(xk + 1 / L * pk) < f(xk) + grad(xk) @ (1 / L * pk) + L / 2 * np.linalg.norm(1 / L * pk):
    while f(xk + pk) < f(xk) + grad(xk) @ pk + L / 2 * np.linalg.norm(pk):
        L /= 2
    return 1 / L


# TODO: сделать такой же формат вывода, как и у scipy_line_search
def line_search(oracle, x_k, p_k=None, method='wolf', tol=1e-3):
    p_k = p_k if p_k is not None else -oracle.grad(x_k)
    if method == 'brent' or method == 'gs':
        l, r = 0, 100
        f_line = lambda x: oracle.value(x_k + x * p_k)

        if method == 'brent':
            return brent(f_line, l, r, eps=tol), 0
        if method == 'gs':
            return golden_section(f_line, l, r, eps=tol), 0
    if method == 'wolf':
        return scipy_line_search(oracle.value, oracle.grad, x_k, p_k)
    if method == 'armijo':
        return armijo(oracle.value, oracle.grad, x_k, p_k)
    if method == 'lip':
        return lip_const(oracle.value, oracle.grad, x_k, p_k), 1


def gradient_descent(oracle, x0, line_search_method='brent', tol=1e-8, max_iter=int(1e4)):
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
        x_k = x_k + alpha * p_k

        if stop_criterion(x_k, tol):
            break
        print(f"{iters}: {oracle.value(x_k)}; a: {alpha}")
        iters += 1
    return x_k, iters


def newton(oracle, x0, line_search_method='brent', tol=1e-8, max_iter=int(1e4)):
    iters = 0

    def stop_criterion(x, tol):
        return np.linalg.norm(x) ** 2 < tol

    x_k = x0.copy()
    while iters < max_iter:
        grad = oracle.grad(x_k)
        hess = oracle.hessian(x_k)
        hess_inv = sparse_inv(hess)
        p_k = -hess_inv @ grad

        alpha = line_search(oracle, x_k, p_k, method=line_search_method, tol=1e-3)[0]
        x_k = x_k + alpha * p_k

        if stop_criterion(x_k, tol):
            break
        print(f"{iters}: {oracle.value(x_k)}; a: {alpha}")
        iters += 1
    return x_k, iters
