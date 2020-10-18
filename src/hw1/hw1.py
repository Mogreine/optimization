import numpy as np


def optimize(oracle, a: float, b: float, eps: float = 1e-8):
    K = 0.381966
    x, w, v = a + K * (b - a), a + K * (b - a), a + K * (b - a)
    d, d2 = 0, 0
    fx = oracle(x)[0]
    fw, fv = fx, fx
    it = []

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
    return np.array(x)
