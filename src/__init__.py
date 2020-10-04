import numpy as np
from scipy.constants import golden_ratio

fi = golden_ratio


def f(x):
    return x * x / 2, x


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


if __name__ == '__main__':
    min_arg, it = golden_section(f, -1e8, 1e8)
    print(f'x0 = {min_arg:.8f}, iterations: {it}')
