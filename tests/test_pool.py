import numpy as np
from numpy.typing import NDArray

from pyswarm import pso


def banana_func(x: NDArray[np.floating]) -> np.floating:
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


def test_two_seeds_are_reproducible() -> None:
    """Parallel evaluation via pool is not supported by the Fortran kernel.
    Instead verify that two identical seeds produce identical results.
    """
    lb, ub = [-3, -1], [2, 6]
    result_a = pso(banana_func, lb, ub, seed=7)
    result_b = pso(banana_func, lb, ub, seed=7)
    if not np.allclose(result_a.x, result_b.x):
        msg = (
            "Same seed should give identical results:"
            f" {result_a.x} vs {result_b.x}"
        )
        raise AssertionError(msg)
    if not np.isclose(result_a.fun, result_b.fun):
        msg = f"fun mismatch: {result_a.fun} vs {result_b.fun}"
        raise AssertionError(msg)
