import numpy as np
import cmath
import random as rnd
import time
from src.hw1.hw1 import optimize
from scipy.constants import golden_ratio


fi = golden_ratio
cos = np.cos
sin = np.sin
ln = np.log
sqrt = np.sqrt


def f0(x: float):
    """f0"""
    return x ** 2 + 1 / 2 * x + 10, 0


def f1(x: float):
    """f1"""
    return x * x / 2 + 2 * x - 3, x


def f2(x: float):
    """f2"""
    return sin(x) + sin(10 / 3 * x), 0


def f4(x: float):
    """f4"""
    return -(16 * x ** 2 - 24 * x + 5) * np.e ** (-x), 0


def f5(x: float):
    """f5"""
    return -(1.4 - 3 * x) * sin(18 * x), 0


def f6(x: float):
    """f6"""
    return -(x + sin(x)) * np.e ** (-(x ** 2)), 0


def f7(x: float):
    """f7"""
    return sin(x) + sin(10 / 3 * x) + ln(x) - 0.84 * x + 3, 0


# Can't calculate the function properly
def f8(x: float):
    """f8"""
    return -sum([k * cos((k + 1) * x + k) for k in range(1, 7)])


def f9(x: float):
    """f9"""
    return sin(x) + sin(2 / 3 * x), 0


def f10(x: float):
    """f10"""
    return -x * sin(x), 0


def f12(x: float):
    """f12"""
    return sin(x) ** 3 + cos(x) ** 3, 0


def f13(x: float):
    """f13"""
    return -(x ** (2 / 3)) - (1 - x ** 2) ** (1 / 3), 0


def golden_section(oracle, a: float, b: float, eps: float = 1e-8):
    l, r = a, b
    x1 = l + (r - l) / (fi + 1)
    x2 = r - (r - l) / (fi + 1)
    f1 = oracle(x1)[0]
    f2 = oracle(x2)[0]
    iterations = 0
    while abs(r - l) > eps:
        if f1 < f2:
            r = x2
            f2 = f1
            x2 = x1
            x1 = l + (r - l) / (fi + 1)
            f1 = oracle(x1)[0]
        else:
            l = x1
            f1 = f2
            x1 = x2
            x2 = r - (r - l) / (fi + 1)
            f2 = oracle(x2)[0]
        iterations += 1
    return (r + l) / 2, iterations


def parabola(oracle, l: float, r: float, eps: float = 1e-8):
    x1 = l
    x3 = r
    # x2 = rnd.uniform(l + 1, r - 1)
    x2 = (l + r) / 2 + 1e-8
    f1 = oracle(x1)[0]
    f2 = oracle(x2)[0]
    f3 = oracle(x3)[0]
    u = x1
    u_prev = x3
    it = 0
    while abs(u_prev - u) > eps:
        u_prev = u
        u = x2 - ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / \
            (2 * (x2 - x1) * (f2 - f3) - 2 * (x2 - x3) * (f2 - f1) + 1e-8)
        # print(u)
        fu = oracle(u)[0]
        if u > x2:
            if fu > f2:
                x3 = u
                f3 = fu
            else:
                x1 = x2
                f1 = f2
                x2 = u
                f2 = fu
        else:
            if fu > f2:
                x1 = u
                f1 = fu
            else:
                x3 = x2
                f3 = f2
                x2 = u
                f2 = fu
        it += 1
    return u, it


def brent(oracle, l: float, r: float, eps: float = 1e-8):
    a = l
    b = r
    x = (l + r) / 2 + 1e-8
    w, v = x, x
    fx = oracle(x)[0]
    fw, fv = fx, fx
    # fa = oracle(a)[0]
    # fb = oracle(b)[0]
    d, d_prev = b - a, b - a
    it = 0
    K = (3 - sqrt(5)) / 2
    for i in range(0, 100):
        d_pr_prev, d_prev = d_prev, d
        tol = eps * abs(x) + eps / 10

        u = x - ((x - w) ** 2 * (fx - fv) - (x - v) ** 2 * (fx - fw)) / \
            (2 * (x - w) * (fx - fv) - 2 * (x - v) * (fx - fw) + 1e-8)

        if u - a < 2 * tol or b - u < 2 * tol:
            u = x - np.sign(x - (a + b) / 2) * tol

        if abs(x - (a + b) / 2) + (b - a) / 2 < 2 * tol:
            break

        # golden section chooses 'u' if parabola is failing
        if not (a < u < b) or abs(a - u) < eps or abs(b - u) < eps or abs(x - v) > d_pr_prev / 2:
            if x < (a + b) / 2:
                u = x + K * (b - a)
                d_prev = b - x
            else:
                u = x - K * (x - a)
                d_prev = x - a
        if abs(u - x) < tol:
            u = x + np.sign(u - x) * tol
        d = abs(u - x)

        fu = oracle(u)[0]
        if fu < fx:
            if u > x:
                a = x
            else:
                b = x
            v = w, w = x, x = u, fv = fw, fw = fx, fx = fu
        else:
            if u > x:
                b = u
            else:
                a = u
            if fu < fw or abs(w - x) < eps:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu < fv or abs(v - x) < eps or abs(v - w) < eps:
                v = u
                fv = fu
        it += 1
    return x, it


# def eq(a: float, b: float, eps: float = 1e-8):
#     return abs(a - b) < eps


def brent_real(oracle, a: float, b: float, eps: float = 1e-8):
    # K = (3 - sqrt(5)) / 2
    K = 0.381966
    # x, w, v = (a + b) / 2, (a + b) / 2, (a + b) / 2
    x, w, v = a + K * (b - a), a + K * (b - a), a + K * (b - a)
    d, d2 = 0, 0
    # d, d2 = K * (r - l), K * (r - l)
    fx = oracle(x)[0]
    fw, fv = fx, fx
    it = []

    pr = 0
    gl = 0
    for i in range(50):
        mid = (a + b) / 2
        tol1 = eps * abs(x) + eps / 10
        tol2 = tol1 * 2
        if abs(x - mid) < tol2 - (b - a) / 2:
            break
        # if len(it) > 0 and abs(x - w) < tol or abs(l - r) < eps:
        #     break
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
            # if eq(x, w) or eq(x, v) or eq(w, v):
            #     q = 0
            r = d2
            d2 = d
        if abs(p) < abs(0.5 * q * r) and q * (a - x) < p < q * (b - x):
            # parabolic interpolation
            d = p / q
            u = x + d
            # f should not be evaluated close to a and b
            if u - a < tol2 or b - u < tol2:
                d = tol1 if x < mid else -tol1
            pr += 1
        else:
            # golden section step
            d2 = (b if x < mid else a) - x
            d = K * d2
            gl += 1
        # f should not be evaluated close to x
        u = x
        if abs(d) > tol1:
            u += d
        elif d > 0:
            u += tol1
        else:
            u -= tol1

        fu = oracle(u)[0]

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
        it.append(x)
    return x, len(it)


def test(method, funcs, x_true, bounds, method_name):
    print(f'Method: {method_name}')
    for i in range(len(funcs)):
        min_arg, it = method(funcs[i], *bounds[i])
        error_arg = abs(min_arg - x_true[i])
        f_true = funcs[i](x_true[i])[0]
        f0 = funcs[i](min_arg)[0]
        error = abs(f0 - f_true)
        print(f'{funcs[i].__name__}, error: {error:.8f}, x_true: {x_true[i]:.6f}, x0: {min_arg:.6f},'
              f' f_true: {f_true:.6f}, f0: {f0:.6f}, iterations: {it}')


if __name__ == '__main__':
    # test(parabola, [f0], [0], [(-4, 40)], 'asd')
    fcs = [f1, f2, f4, f5, f6, f7, f10, f12, f13, f9]
    ans = [-2, 5.145735, 2.868034, 0.96609, 0.67956, 5.19978, 7.9787, np.pi, 1 / np.sqrt(2), 17.039]
    bounds = [(-6, 2), (2.7, 7.5), (1.9, 3.9), (0, 1.2), (-10, 10), (2.7, 7.5), (0, 10), (0, 2 * np.pi), (0, 1),
              (3.1, 20.4)]
    # print(f5(ans[3]))
    test(golden_section, fcs, ans, bounds, 'golden search')
    test(parabola, fcs, ans, bounds, 'parabola')
    test(optimize, fcs, ans, bounds, 'brent')
