import numpy as np
from numpy.typing import NDArray

from pyswarm import pso


def banana_func(x: NDArray[np.floating]) -> np.floating:
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


def test_scipy_bounds_matches_lb_ub() -> None:
    bounds = [(-3, 2), (-1, 6)]
    result_b    = pso(banana_func, bounds=bounds, seed=42)
    result_lbub = pso(banana_func, lb=[-3, -1], ub=[2, 6], seed=42)
    xopt_b,    fopt_b    = result_b.x,    result_b.fun
    xopt_lbub, fopt_lbub = result_lbub.x, result_lbub.fun
    if not np.allclose(xopt_b, xopt_lbub):
        msg = (
            "bounds and lb/ub should produce identical results:"
            f" {xopt_b} vs {xopt_lbub}"
        )
        raise AssertionError(msg)
    if not np.isclose(fopt_b, fopt_lbub):
        msg = f"fopt mismatch: {fopt_b} vs {fopt_lbub}"
        raise AssertionError(msg)
