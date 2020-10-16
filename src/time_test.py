import numpy as np
import cmath
import os
import math
import profile
import random
from scipy.constants import golden_ratio

fi = golden_ratio
cos = np.cos
sin = np.sin
ln = np.log

N = int(1e4)

arr = [i for i in range(N)]
args = arr.copy()

rand = lambda: int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)


def run_cos():
    global N
    for i in range(N):
        arr[i] = cos(rand())


if __name__ == '__main__':
    # print(rand())
    profile.run('print(run_cos())')
    print(arr)
