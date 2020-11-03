import numpy as np
import scipy
from src.hw2.oracle import make_oracle, OracleTester
from src.hw2.optimization import GradientDescent


a = np.array([1, 2, 3])
b = np.array([1, 1, 1])

oracle = make_oracle('hw2/data/a1a.txt')

w_0 = np.random.normal(0, 1, oracle.N)

w_opt = GradientDescent.optimize(oracle, w_0)

print(w_opt)



