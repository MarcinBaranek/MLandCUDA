from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np

from config import D
from core import get_position
from paths import euler_last_point
from user import a, b


def get_kernel_weak_approx(a_func, b_func):
    @cuda.jit
    def kernel(initial_point, t_0, T, n, state, result):
        pos = get_position()
        if pos < result.size / initial_point.shape[0]:
            tmp = cuda.local.array(shape=(D, 1), dtype=initial_point.dtype)
            euler_last_point(
                initial_point, t_0, T, n, a_func, b_func, state, tmp
            )
            for j in range(D):
                result[pos, j] = tmp[j, 0]

    return kernel


N = 1024
point = np.ones(shape=(D, 1)).astype('float32')
result = np.zeros(shape=(N, D)).astype('float32')

d_point = cuda.to_device(point)
d_result = cuda.to_device(result)
state = create_xoroshiro128p_states(N, seed=2)

kernel = get_kernel_weak_approx(a, b)
kernel[N // 8, 8](d_point, 0, 1, 100, state, d_result)

d_result.copy_to_host(result)
print(result.mean(axis=0))  # [1.020957  1.1317459]
