import math

import numba.cuda as cuda


@cuda.jit(device=True)
def a(t, x, result):
    result[0, 0] = 0.5 * t * math.sin(10*x[0, 0])
    result[1, 0] = 0.5 * math.cos(7 * x[1, 0])


@cuda.jit(device=True)
def b(t, x, result):
    result[0, 0] = t * x[0, 0]
    result[0, 1] = t * x[1, 0]
    result[0, 2] = math.sin(x[1, 0])
    result[1, 0] = math.cos(x[0, 0])
    result[1, 1] = x[1, 0]
    result[1, 2] = -x[0, 0]