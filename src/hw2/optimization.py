import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import eig as scipy_eig, cho_solve, cho_factor, solve_triangular
from scipy.sparse.linalg import inv as sparse_inv
from scipy.optimize import line_search as scipy_line_search, brent as brent_sc, bracket
from scipy.sparse.linalg import norm, svds, eigsh
import time


class LineSearchOptimizer:
    def __call__(self, *args, **kwargs):
        pass


class GoldenSection(LineSearchOptimizer):
    def __init__(self, tol, max_iter, bracketing=True):
        self.tol = tol
        self.max_iter = max_iter
        self.bracketing = bracketing

    def __call__(self, *args, **kwargs):
        f_line = lambda x: kwargs['f'](kwargs['x_k'] + x * kwargs['p_k'])
        oracle_calls = 0
        if self.bracketing:
            l, r = 0, 100
            xa, xb, xc, fa, fb, fc, calls = bracket(f_line, xa=l, xb=r)
            oracle_calls += calls
            l, r = xa, xc
        else:
            l, r = 0, 100
        alpha, calls = self.golden_section(f_line, l, r, self.tol, self.max_iter)
        return alpha, calls + oracle_calls

    @staticmethod
    def golden_section(func, a: float, b: float, tol: float = 1e-8, max_iter=50):
        l, r = a, b
        K = 0.381966
        x1 = l + K * (r - l)
        x2 = r - K * (r - l)
        f1 = func(x1)
        f2 = func(x2)
        iterations = 0
        while abs(r - l) > tol and iterations < max_iter:
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


class Brent(LineSearchOptimizer):
    def __init__(self, bracketing=True):
        self.bracketing = bracketing

    def __call__(self, *args, **kwargs):
        f_line = lambda x: kwargs['f'](kwargs['x_k'] + x * kwargs['p_k'])
        oracle_calls = 0
        if self.bracketing:
            l, r = 0, 100
            xa, xb, xc, fa, fb, fc, calls = bracket(f_line, xa=l, xb=r)
            oracle_calls += calls
            brack = (xa, xb, xc)
        else:
            brack = (0, 100)
        alpha, _, _, calls = brent_sc(f_line, brack=brack, tol=kwargs['tol'], full_output=True)
        return alpha, calls + oracle_calls


class Armijo(LineSearchOptimizer):
    def __init__(self, po=0.1, c=0.0001):
        self.po = po
        self.c = c

    def __call__(self, *args, **kwargs):
        alpha, calls = self.armijo(kwargs['f'], kwargs['f_grad'], kwargs['x_k'], kwargs['p_k'],
                                   self.po, self.c, kwargs['is_newton'])
        return alpha, calls

    @staticmethod
    def armijo(f, grad, xk, pk, po=0.1, c=0.0001, is_newton=False):
        alpha = 1 if is_newton else 100
        xk_grad = grad(xk)
        xk_val = f(xk)
        oracle_calls = 2

        # armijo condition
        while f(xk + alpha * pk) > xk_val + c * alpha * xk_grad @ pk:
            alpha *= po
            oracle_calls += 1

        return alpha, oracle_calls


class Wolfe(LineSearchOptimizer):
    def __init__(self, c1=0.0001, c2=0.9):
        self.c1 = c1
        self.c2 = c2

    def __call__(self, *args, **kwargs):
        alpha, value_calls, grad_calls = scipy_line_search(kwargs['f'], kwargs['f_grad'],
                                                           kwargs['x_k'], kwargs['p_k'], c1=self.c1, c2=self.c2)[:3]
        return alpha, value_calls + grad_calls


class Lipschitz(LineSearchOptimizer):
    def __call__(self, *args, **kwargs):
        return self.lip_const(kwargs['f'], kwargs['f_grad'], kwargs['x_k'], kwargs['p_k'])

    @staticmethod
    def lip_const(f, grad, xk, pk):
        L = 0.01
        xk_grad = grad(xk)
        xk_val = f(xk)
        oracle_calls = 2

        while f(xk + 1 / L * pk) > xk_val + 1 / L * xk_grad @ pk + 1 / 2 / L * pk @ pk:
            L *= 2
            oracle_calls += 1

        return 2 / L, oracle_calls


class Wolfe_Arm(LineSearchOptimizer):
    def __init__(self):
        self.wolfe = Wolfe()
        self.armijo = Armijo()

    def __call__(self, *args, **kwargs):
        alpha, wolfe_calls = self.wolfe(*args, **kwargs)
        if alpha is None:
            alpha, armijo_calls = self.armijo(*args, **kwargs)
            return alpha, armijo_calls + wolfe_calls
        return alpha, wolfe_calls


class Optimizer:
    def __init__(self, oracle):
        self.oracle = oracle
        self.oracle_calls_arr = [0]
        self.elapsed_time_arr = [0]
        self.iters_arr = [0]
        self.rk_arr = [0]
        self.start_time = 0

    def init_data(self, x0):
        self.start_time = time.time()
        self.oracle_calls_arr = [0]
        self.elapsed_time_arr = [0]
        self.iters_arr = [0]
        self.rk_arr = [self.oracle.value(x0)]

    def log_data(self, oracle_calls, x_k):
        self.elapsed_time_arr.append(time.time() - self.start_time)
        self.oracle_calls_arr.append(self.oracle_calls_arr[-1] + 1 + oracle_calls)
        self.iters_arr.append(self.iters_arr[-1] + 1)
        self.rk_arr.append(self.oracle.value(x_k))

    def get_data(self):
        return self.rk_arr, self.elapsed_time_arr, self.oracle_calls_arr, self.iters_arr

    def optimize(self, x0, line_search, tol=1e-8, max_iter=int(1e4), verbose=0) -> np.ndarray:
        pass

    def stop_criterion(self, x0_grad, x_grad, tol):
        pass


class GradientDescent(Optimizer):
    def stop_criterion(self, x0_grad, x_grad, tol):
        x_grad_norm = np.linalg.norm(x_grad)
        x0_grad_norm = np.linalg.norm(x0_grad)
        rk = x_grad_norm ** 2 / x0_grad_norm ** 2
        return rk < tol

    def optimize(self, x0, line_search, tol=1e-8, max_iter=int(1e4), verbose=0) -> np.ndarray:
        x0_grad = self.oracle.grad(x0)
        x_k = x0.copy()
        self.init_data(x0)
        for i in range(max_iter):
            grad = self.oracle.grad(x_k)
            p_k = -grad

            alpha, oracle_calls = line_search(f=self.oracle.value, f_grad=self.oracle.grad, x_k=x_k, p_k=p_k, tol=1e-8,
                                              is_newton=False)
            if alpha is None:
                alpha = 1

            x_k = x_k + alpha * p_k

            self.log_data(oracle_calls + 1, x_k)

            if self.stop_criterion(x0_grad, grad, tol):
                break
            if verbose == 1:
                print(f"{self.iters_arr[-1]}: {self.oracle.value(x_k)}; a: {alpha}")
        return x_k


class Newton(Optimizer):
    def __init__(self, oracle, solve='cholesky'):
        super().__init__(oracle)
        self.solve = solve

    def is_positive(self, X):
        try:
            L = np.linalg.cholesky(X)
        except np.linalg.LinAlgError:
            return False
        return True

    def is_symmetric(self, X):
        return np.allclose(X, X.T)

    def is_positive_symmetric(self, X):
        return self.is_symmetric(X) and self.is_positive(X)

    def correct_hessian_addition(self, H):
        eps = 1e-14
        I = np.eye(H.shape[0]) * eps
        while not self.is_positive(H + I):
            I *= 2
        return H + I

    # only for p.d. matrices
    def solve_le_cholesky(self, X, b):
        L = cho_factor(X, check_finite=False)
        x = cho_solve(L, b, check_finite=False)
        return x

    def solve_le_inv(self, X, b):
        return np.linalg.inv(X) @ b

    def solve_conj(self, A, b, Ax=None, tol=1e-8):
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

    def solve_le(self, X, b, method='cholesky'):
        if method == 'cholesky':
            return self.solve_le_cholesky(X, b)
        if method == 'inv':
            return self.solve_le_inv(X, b)
        if method == 'conj':
            return self.solve_conj(X, b)

    def stop_criterion(self, x0_grad, x_grad, tol):
        x_grad_norm = np.linalg.norm(x_grad)
        x0_grad_norm = np.linalg.norm(x0_grad)
        rk = x_grad_norm ** 2 / x0_grad_norm ** 2
        return rk < tol

    def stop_criterion2(self, x0_grad, x_grad, tol):
        return np.linalg.norm(x_grad) ** 2 < tol

    def optimize(self, x0, line_search, tol=1e-8, max_iter=int(1e4), verbose=0) -> np.ndarray:
        x0_grad = self.oracle.grad(x0)
        x_k = x0.copy()
        self.init_data(x0)

        for i in range(max_iter):
            grad, hess = self.oracle.fuse_grad_hessian(x_k)

            if self.stop_criterion2(x0_grad, grad, tol):
                break

            hess = self.correct_hessian_addition(hess)
            p_k = -self.solve_le(hess, grad, method=self.solve)

            if np.linalg.norm(p_k) > 1000:
                p_k = p_k / np.linalg.norm(p_k)

            alpha, oracle_calls = line_search(f=self.oracle.value, f_grad=self.oracle.grad, x_k=x_k, p_k=p_k, tol=1e-8,
                                              is_newton=True)
            if alpha is None:
                alpha = 1

            x_k = x_k + alpha * p_k

            if verbose == 1:
                print(f"{self.iters_arr[-1]}: {self.oracle.value(x_k)}; a: {alpha}")

            self.log_data(oracle_calls, x_k)
        return x_k


class HFN(Optimizer):
    def stop_criterion(self, x0_grad, x_grad, tol):
        return np.linalg.norm(x_grad) ** 2 < tol

    def solve_conj_hess_free_line_search(self, Hx, grad, tol=1e-20):
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

    def optimize(self, x0, line_search, tol=1e-8, max_iter=int(1e4), verbose=0) -> np.ndarray:
        x0_grad = self.oracle.grad(x0)
        x_k = x0.copy()
        self.init_data(x0)

        for i in range(max_iter):
            grad = self.oracle.grad(x_k)

            if self.stop_criterion(x0_grad, grad, tol):
                break

            Hd = lambda d: self.oracle.hessian_vec_product(x_k, d)
            p_k, hess_vec_prod_calls = self.solve_conj_hess_free_line_search(Hd, grad)

            if np.linalg.norm(p_k) > 1000:
                p_k = p_k / np.linalg.norm(p_k)

            alpha, oracle_calls = line_search(f=self.oracle.value, f_grad=self.oracle.grad, x_k=x_k, p_k=p_k, tol=1e-8,
                                              is_newton=True)
            if alpha is None:
                alpha = 1

            x_k = x_k + alpha * p_k

            if verbose == 1:
                print(f"{self.iters_arr[-1]}: {self.oracle.value(x_k)}; a: {alpha}")
            self.log_data(oracle_calls + hess_vec_prod_calls + 1, x_k)
        return x_k


class BFGS(Optimizer):
    def stop_criterion(self, x0_grad, x_grad, tol):
        return np.linalg.norm(x_grad) ** 2 < tol

    def recalc(self, B, s, y, *args):
        id = args[0]
        po = 1 / (s @ y)
        M = id - po * np.outer(s, y)
        B = M @ B @ M.T + po * np.outer(s, s)
        return B

    def optimize(self, x0, line_search, tol=1e-8, max_iter=int(1e4), verbose=0) -> np.ndarray:
        g = self.oracle.grad(x0)
        g_pr = g.copy()

        x_pr = x0.copy()
        x = x0.copy()

        B = np.eye(self.oracle.features) * 1
        id = np.eye(self.oracle.features)

        x0_grad = self.oracle.grad(x0)
        self.init_data(x0)

        for i in range(max_iter):
            if self.stop_criterion(x0_grad, g, tol):
                break

            assert isinstance(g, np.ndarray)
            pk = -B @ g

            alpha, oracle_calls = line_search(f=self.oracle.value, f_grad=self.oracle.grad, x_k=x, p_k=pk, tol=1e-8,
                                              is_newton=True)
            if alpha is None:
                alpha = 1
            x = x + alpha * pk
            g = self.oracle.grad(x)

            s = x - x_pr
            y = g - g_pr

            B = self.recalc(B, s, y, id)

            g_pr = g.copy()
            x_pr = x.copy()

            if alpha is None:
                alpha = 1

            if verbose == 1:
                print(f"{self.iters_arr[-1]}: {self.oracle.value(x)}; a: {alpha}")
            self.log_data(oracle_calls + 1, x)
        return x


class LBFGS(Optimizer):
    def __init__(self, oracle, m: int):
        super().__init__(oracle)
        self.m = m
        self.n = oracle.features
        self.S = np.ndarray
        self.Y = np.ndarray
        self.R = np.zeros(shape=(m, m))
        self.D = np.zeros(shape=(m, m))

    def stop_criterion(self, x0_grad, x_grad, tol):
        return np.linalg.norm(x_grad) ** 2 < tol

    def recalc(self, s: np.ndarray, y: np.ndarray, g: np.ndarray):
        R = self.R
        S = self.S
        Y = self.Y
        D = self.D

        Y[:, :-1] = Y[:, 1:]
        Y[:, -1] = y

        S[:, :-1] = S[:, 1:]
        S[:, -1] = s

        R[:-1, :-1] = R[1:, 1:]
        R[-1] = 0
        R[:, -1] = S.T @ y

        R_inv = np.linalg.inv(R)

        D[:-1, :-1] = D[1:, 1:]
        D[-1, -1] = s @ y

        Y_qd = self.Y.T @ self.Y
        gamma = y @ s / (y @ y)

        p = np.vstack((R_inv.T @ (D + gamma * Y_qd) @ R_inv @ (S.T @ g) - gamma * R_inv.T @ (Y.T @ g),
                       -R_inv @ (S.T @ g))).reshape(-1, 1)

        Hg = gamma * g + (np.hstack((S, gamma * Y)) @ p).flatten()

        self.R = R
        self.S = S
        self.Y = Y
        self.D = D
        return Hg

    def recalc2(self, s: np.ndarray, y: np.ndarray, g: np.ndarray, k):
        R = self.R
        S = self.S
        Y = self.Y
        D = self.D

        if k == 1:
            S = s.reshape(-1, 1)
            Y = y.reshape(-1, 1)
            R = np.zeros((1, 1)) + s @ y
            D = np.zeros((1, 1)) + s @ y
        else:
            S = np.hstack((S, s.reshape(-1, 1)))
            Y = np.hstack((Y, y.reshape(-1, 1)))

            # expand R
            tmp = np.zeros(shape=(k, k))
            tmp[:-1, :-1] = R
            R = tmp
            R[:, -1] = S.T @ y

            D = np.diagflat(np.append(np.diagonal(D), s @ y))

        R_inv = np.linalg.inv(R)
        Y_qd = Y.T @ Y
        gamma = y @ s / (y @ y)

        p = np.vstack((R_inv.T @ (D + gamma * Y_qd) @ R_inv @ (S.T @ g) - gamma * R_inv.T @ (Y.T @ g),
                       -R_inv @ (S.T @ g))).reshape(-1, 1)

        Hg = gamma * g + (np.hstack((S, gamma * Y)) @ p).flatten()


        self.R = R
        self.S = S
        self.Y = Y
        self.D = D
        return Hg

    def optimize(self, x0, line_search, tol=1e-8, max_iter=int(1e4), verbose=0) -> np.ndarray:
        g = self.oracle.grad(x0)
        g_pr = g.copy()

        x_pr = x0.copy()
        x = x0.copy()

        x0_grad = self.oracle.grad(x0)
        self.init_data(x0)

        pk = -g

        for i in range(max_iter):
            if self.stop_criterion(x0_grad, g, tol):
                break

            alpha, oracle_calls = line_search(f=self.oracle.value, f_grad=self.oracle.grad, x_k=x, p_k=pk, tol=1e-8,
                                              is_newton=True)
            if alpha is None:
                alpha = 1
            x = x + alpha * pk
            g = self.oracle.grad(x)

            s = x - x_pr
            y = g - g_pr

            if i < self.m:
                pk = -self.recalc2(s, y, g, i + 1)
            else:
                pk = -self.recalc(s, y, g)

            g_pr = g.copy()
            x_pr = x.copy()

            if verbose == 1:
                print(f"{self.iters_arr[-1]}: {self.oracle.value(x)}; a: {alpha}")
            self.log_data(oracle_calls + 1, x)
        return x


class LogRegl1(Optimizer):
    def __init__(self, oracle, l: float):
        super().__init__(oracle)
        self.l = l

    def stop_criterion(self, xk_prev, xk, alpha, tol):
        return np.linalg.norm((xk - xk_prev) / alpha) ** 2 < tol

    def prox_operator(self, x, grad, alpha):
        alpha = 1 / alpha
        pk_old = x - alpha * grad
        max_part = np.absolute(pk_old) - alpha * self.l
        max_part[max_part < 0] = 0
        return np.sign(pk_old) * max_part

    def line_search(self, xk, xk_grad):
        oracle_calls = 1
        xk_val = self.oracle.value(xk)
        L = 0.01
        x = self.prox_operator(xk, xk_grad, L)
        F = lambda x: self.oracle.value(x) + self.l * np.linalg.norm(x, ord=1)

        while F(x) > xk_val + xk_grad @ (x - xk) + L / 2 * np.linalg.norm(x - xk) ** 2 + self.l * np.linalg.norm(x, ord=1):
            L *= 2
            x = self.prox_operator(xk, xk_grad, L)
            oracle_calls += 1
        L /= 2
        x = self.prox_operator(xk, xk_grad, L)
        return x, oracle_calls, 1 / L

    def optimize(self, x0, line_search=None, tol=1e-8, max_iter=int(1e4), verbose=0) -> np.ndarray:
        xk = x0.copy()
        self.init_data(x0)
        for i in range(max_iter):
            grad = self.oracle.grad(xk)

            xk_prev = xk.copy()
            xk, oracle_calls, alpha = self.line_search(xk, grad)

            self.log_data(oracle_calls + 1, xk)

            if self.stop_criterion(xk_prev, xk, alpha, tol):
                break
            if verbose == 1:
                print(f"{self.iters_arr[-1]}: {self.oracle.value(xk)}; a: {alpha}")
        return xk
