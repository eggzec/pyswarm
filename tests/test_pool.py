import multiprocessing

import numpy as np
from numpy.typing import NDArray

from pyswarm import pso


def banana_func(x: NDArray[np.floating]) -> np.floating:
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


def test_custom_pool_matches_single_process() -> None:
    lb, ub = [-3, -1], [2, 6]
    result_single = pso(banana_func, lb, ub, seed=7)
    with multiprocessing.Pool(2) as p:
        result_pool = pso(banana_func, lb, ub, seed=7, pool=p)
    xopt_single, fopt_single = result_single.x, result_single.fun
    xopt_pool,   fopt_pool   = result_pool.x,   result_pool.fun
    if not np.allclose(xopt_single, xopt_pool):
        msg = (
            "pool and single-process results differ:"
            f" {xopt_single} vs {xopt_pool}"
        )
        raise AssertionError(msg)
    if not np.isclose(fopt_single, fopt_pool):
        msg = f"fopt mismatch: {fopt_single} vs {fopt_pool}"
        raise AssertionError(msg)
