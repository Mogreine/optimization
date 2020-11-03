import numpy as np
import scipy


class GradientDescent:
    @staticmethod
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

    @staticmethod
    def optimize(oracle, x0, line_search_method='gs', tol=1e-8, max_iter=int(1e5)):
        line_search = GradientDescent.golden_section
        iters = 0

        def stop_criterion(x, y, tol):
            tmp = np.linalg.norm(x) ** 2 / np.linalg.norm(y) ** 2
            return tmp < tol

        x_k = x0
        while iters < max_iter:
            grad = oracle.grad(x_k)
            p_k = grad / np.linalg.norm(grad)
            f_line = lambda x: oracle.value(x_k - x * p_k)

            # bracketing
            l, r = 0, 1 / 2 ** (-16)
            # Goldstein condition for bracketing
            f_k = oracle.value(x_k)
            while oracle.value(x_k - r * p_k) < f_k + r * grad @ p_k:
                r *= 2

            alpha = line_search(f_line, 0, r)
            x_k = x_k - alpha * grad

            if stop_criterion(x_k, x0, tol):
                break
            print(oracle.value(x_k))
        return x_k

