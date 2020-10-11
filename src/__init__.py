import numpy as np
import cmath
import random as rnd
from scipy.constants import golden_ratio

fi = golden_ratio
cos = np.cos
sin = np.sin
ln = np.log


def f0(x: float):
    return x ** 2 + 1 / 2 * x + 10, 0


def f1(x: float):
    return x * x / 2, x


def f2(x: float):
    return sin(x) + sin(10 / 3 * x), 0


def f4(x: float):
    return -(16 * x ** 2 - 24 * x + 5) * np.e ** (-x), 0


def f5(x: float):
    return -(1.4 - 3 * x) * sin(18 * x), 0


def f6(x: float):
    return -(x + sin(x)) * np.e ** (-(x ** 2)), 0


def f7(x: float):
    return sin(x) + sin(10 / 3 * x) + ln(x) - 0.84 * x + 3, 0


# Can't calculate the function properly
def f8(x: float):
    return -sum([k * cos((k + 1) * x + k) for k in range(1, 7)])


def f9(x: float):
    return sin(x) + sin(2 / 3 * x), 0


def f10(x: float):
    return -x * sin(x), 0


def f12(x: float):
    return sin(x) ** 3 + cos(x) ** 3, 0


def f13(x: float):
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
        u = x2 - ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) /\
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


# def brent(oracle, l: float, r: float, eps: float = 1e-8):
#     K = (3 - np.sqrt(5)) / 2
#     x, x_prev, mid = 0, 0, 0
#     a, b, c = 0, 0, 0
#     fa, fb, fc = 0, 0, 0
#     spare = 0
#     while abs(a - c) < eps:
#         # Если мы ошиблись уже 2 раза выполним golden_search
#         if spare > 1:
#             x1 = a + K * (c - a)
#             x2 = c - K * (c - a)
#             f1 = oracle(x1)[0]
#             f2 = oracle(x2)[0]
#             if f1 < f2:
#                 r = x2
#             else:
#                 l = x1
#             spare = 0
#             continue
#         x_prev = x
#         x=mid -


def test(method, funcs, x_true, bounds, method_name):
    print(f'Method: {method_name}')
    for i in range(len(funcs)):
        min_arg, it = method(funcs[i], *bounds[i])
        error = abs(min_arg - x_true[i])
        f_true = funcs[i](x_true[i])[0]
        f0 = funcs[i](min_arg)[0]
        error = abs(f0 - f_true)
        print(f'f{i + 1}, error: {error:.8f}, x_true: {x_true[i]:.6f}, x0: {min_arg:.6f},'
              f' f_true: {f_true:.6f}, f0: {f0:.6f}, iterations: {it}')


if __name__ == '__main__':
    # test(parabola, [f0], [0], [(-4, 40)], 'asd')
    fcs = [f1, f2, f4, f5, f6, f7, f10, f12, f13, f9]
    ans = [0, 5.145735, 2.868034, 0.96609, 0.67956, 5.19978, 7.9787, np.pi, 1 / np.sqrt(2), 17.039]
    bounds = [(-1, 1), (2.7, 7.5), (1.9, 3.9), (0, 1.2), (-10, 10), (2.7, 7.5), (0, 10), (0, 2 * np.pi), (0.001, 0.99), (1, 20.4)]
    # print(f5(ans[3]))
    test(golden_section, fcs, ans, bounds, 'golden search')
    test(parabola, fcs, ans, bounds, 'parabola')
