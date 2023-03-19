from numba import cuda

from core import write_from_to
from steps import euler_step


@cuda.jit(device=True)
def euler_last_point(
        initial_point, t_0, end_time, n, a_func, b_func, state, result
):
    write_from_to(initial_point, result)
    dt = (end_time - t_0) / n
    time = t_0
    while time < end_time:
        euler_step(result, time, dt, a_func, b_func, state)
        time += dt
