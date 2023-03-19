import numpy as np

THREADS = 16
BLOCKS = 8

D = 2
M = 3
K = THREADS * BLOCKS
N = 2


x_0 = np.reshape(np.array([1, 2]), newshape=(D, 1)).astype('float32')
time = np.reshape(np.linspace(0, 2, 100), newshape=(-1, 1)).astype('float32')
