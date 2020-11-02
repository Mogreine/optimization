import numpy as np
import scipy

a = np.array([1, 2, 3])
b = np.array([1, 1, 1])

X = np.array([*range(9)]).reshape((3, 3))
print(np.diagflat(a))
