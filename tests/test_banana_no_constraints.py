import numpy as np
from numpy.typing import NDArray

from pyswarm.pso import pso


def banana_func(x: NDArray[np.floating]) -> np.floating:
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


def test_banana_no_constraints() -> None:
    lb = [-3, -1]
    ub = [2, 6]
    xopt, fopt = pso(banana_func, lb, ub, debug=True)
    if not np.allclose(xopt, [1.0, 1.0], atol=0.001):
        msg = f"Expected xopt close to [1.0, 1.0], got {xopt}"
        raise AssertionError(msg)
    if not np.isclose(fopt, 4.0):
        msg = f"Expected fopt close to 4.0, got {fopt}"
        raise AssertionError(msg)
