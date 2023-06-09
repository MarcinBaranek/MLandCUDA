import numba.cuda as cuda
import math
from numba.cuda.random import xoroshiro128p_normal_float32


@cuda.jit(device=True)
def add(array_a, array_b, result):
    for i in range(array_a.shape[0]):
        for j in range(array_a.shape[1]):
            result[i, j] = array_a[i, j] + array_b[i, j]


@cuda.jit(device=True)
def add_inplace(array_a, array_b):
    for i in range(array_a.shape[0]):
        for j in range(array_a.shape[1]):
            array_a[i, j] += array_b[i, j]


@cuda.jit(device=True)
def minus_inplace(array_a, array_b):
    for i in range(array_a.shape[0]):
        for j in range(array_a.shape[1]):
            array_a[i, j] -= array_b[i, j]


@cuda.jit(device=True)
def multiply_matrix(matrix_a, matrix_b, result):
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_b.shape[1]):
            temp = 0
            for k in range(matrix_a.shape[1]):
                temp += matrix_a[i, k] * matrix_b[k, j]
            result[i, j] = temp


@cuda.jit(device=True)
def multiply_matrix_by_scalar(matrix, scalar):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = matrix[i, j] * scalar


@cuda.jit(device=True)
def norm(vector):
    total = 0
    for i in range(vector.shape[0]):
        for j in range(vector.shape[1]):
            total += vector[i, j] ** 2
    return math.sqrt(total)


@cuda.jit(device=True)
def gen_normal_float32(vector, state):
    thread_id = cuda.grid(1)
    for i in range(vector.shape[0]):
        for j in range(vector.shape[1]):
            vector[i, j] = xoroshiro128p_normal_float32(state, thread_id)


@cuda.jit(device=True)
def write_from_to(source, destination):
    """Just copy source to destination (should have the same shape)."""
    for i in range(destination.shape[0]):
        for j in range(destination.shape[1]):
            destination[i, j] = source[i, j]


@cuda.jit(device=True)
def get_position():
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    return tx + ty * bw


@cuda.jit(device=True)
def fill(vector, value: float):
    for i in range(vector.shape[0]):
        for j in range(vector.shape[1]):
            vector[i, j] = value


@cuda.jit(device=True)
def sse(array_a, array_b):
    total = 0
    for i in range(array_a.shape[0]):
        for j in range(array_a.shape[1]):
            total += (array_a[i, j] - array_b[i, j]) ** 2
    return total
