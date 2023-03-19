import math

from numba import cuda

from core import (
    multiply_matrix_by_scalar, gen_normal_float32, multiply_matrix, add_inplace
)
from config import D, M


# ====================
# === Simple Steps ===
# ====================

@cuda.jit(device=True)
def drift_step(point, time, dt, a_func, result):
    """Perform calculations for drift step.

    Compute
    .. math::
        a(t, x)dt

    Parameters
    ----------
    point
     : Array(D, 1)
        Start point (space variable)
    time : float
    dt : float
    a_func
    result : A
    """
    a_func(time, point, result)
    multiply_matrix_by_scalar(result, dt)


@cuda.jit(device=True)
def diffusion_step(point, time, dt, b_func, result, state):
    temp = cuda.local.array(shape=(D, M), dtype=point.dtype)
    dw = cuda.local.array(shape=(M, 1), dtype=point.dtype)

    gen_normal_float32(dw, state)
    multiply_matrix_by_scalar(dw, math.sqrt(dt))
    b_func(time, point, temp)
    multiply_matrix(temp, dw, result)


@cuda.jit(device=True)
def diffusion_step_with_dw(point, time, dw, b_func, result):
    temp = cuda.local.array(shape=(D, M), dtype=point.dtype)

    b_func(time, point, temp)
    multiply_matrix(temp, dw, result)


# ===========================
# === Inexact information ===
# ===========================

@cuda.jit(device=True)
def drift_step_random_a(point, time, dt, a_func, result, state, delta):
    a_func(time, point, delta, result, state)
    multiply_matrix_by_scalar(result, dt)


@cuda.jit(device=True)
def diffusion_step_with_dw_random_b(point, time, dw, b_func, result, state, delta):
    temp = cuda.local.array(shape=(D, M), dtype=point.dtype)

    b_func(time, point, delta, temp, state)
    multiply_matrix(temp, dw, result)


@cuda.jit(device=True)
def euler_step_with_dw_random_a_b(point, time, xi, dt, a_func, b_func, dw, state, delta):
    temp_drift = cuda.local.array(shape=(D, 1), dtype=point.dtype)
    temp_diffusion = cuda.local.array(shape=(D, 1), dtype=point.dtype)

    drift_step_random_a(point, xi, dt, a_func, temp_drift, state, delta)
    diffusion_step_with_dw_random_b(point, time, dw, b_func, temp_diffusion, state, delta)
    add_inplace(point, temp_drift)
    add_inplace(point, temp_diffusion)


@cuda.jit(device=True)
def euler_step(point, time, dt, a_func, b_func, state) -> None:
    """Perform euler step and write results into the `point`

    Compute
    .. math::
        a(t, x)dt + b(t, x)dW_t
    Parameters
    ----------
    point
        Start point will be overwritten.
    time: float
    dt: float
    a_func
    b_func
    state
        Random state.
    """
    temp_drift = cuda.local.array(shape=(D, 1), dtype=point.dtype)
    temp_diffusion = cuda.local.array(shape=(D, 1), dtype=point.dtype)

    drift_step(point, time, dt, a_func, temp_drift)
    diffusion_step(point, time, dt, b_func, temp_diffusion, state)
    add_inplace(point, temp_drift)
    add_inplace(point, temp_diffusion)


@cuda.jit(device=True)
def euler_step_with_dw(point, time, dt, a_func, b_func, dw) -> None:
    temp_drift = cuda.local.array(shape=(D, 1), dtype=point.dtype)
    temp_diffusion = cuda.local.array(shape=(D, 1), dtype=point.dtype)

    drift_step(point, time, dt, a_func, temp_drift)
    diffusion_step_with_dw(point, time, dw, b_func, temp_diffusion)
    add_inplace(point, temp_drift)
    add_inplace(point, temp_diffusion)


@cuda.jit(device=True)
def randomized_euler_step_with_dw(point, time, xi, dt, a_func, b_func, dw):
    temp_drift = cuda.local.array(shape=(D, 1), dtype=point.dtype)
    temp_diffusion = cuda.local.array(shape=(D, 1), dtype=point.dtype)

    drift_step(point, xi, dt, a_func, temp_drift)
    diffusion_step_with_dw(point, time, dw, b_func, temp_diffusion)
    add_inplace(point, temp_drift)
    add_inplace(point, temp_diffusion)