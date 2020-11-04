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

    @staticmethod
    def optimize(oracle, x0, line_search_method='brent', tol=1e-8, max_iter=int(1e5)):
        line_search = GradientDescent.brent
        if line_search_method == 'gs':
            line_search = GradientDescent.golden_section
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
            p_k = grad / np.linalg.norm(grad)
            f_line = lambda x: oracle.value(x_k - x * grad)

            l, r = 0, 20
            alpha = line_search(f_line, l, r, eps=1e-3)
            x_k = x_k - alpha * grad

            if stop_criterion(x_k, tol):
                break
            print(f"{iters}: {oracle.value(x_k)}; a: {alpha}")
            iters += 1
        return x_k, iters

