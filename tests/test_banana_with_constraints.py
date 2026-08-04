import numpy as np
from numpy.typing import NDArray

from pyswarm import pso


def banana_func(x: NDArray[np.floating]) -> np.floating:
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


def banana_constraint(x: NDArray[np.floating]) -> list[float]:
    x1 = x[0]
    x2 = x[1]
    return [-((x1 + 0.25) ** 2) + 0.75 * x2]


def test_banana_with_constraints() -> None:
    lb = [-3, -1]
    ub = [2, 6]
    result = pso(
        banana_func, lb, ub, f_ieqcons=banana_constraint, debug=True, seed=2
    )
    xopt, fopt = result.x, result.fun
    if not np.allclose(xopt, [0.5, 0.75], atol=0.1):
        msg = f"Expected xopt close to [0.5, 0.75], got {xopt}"
        raise AssertionError(msg)
    if not np.isclose(fopt, 4.5, atol=0.1):
        msg = f"Expected fopt close to 4.5, got {fopt}"
        raise AssertionError(msg)
    constraints = banana_constraint(xopt)
    if not all(c >= 0 for c in constraints):
        msg = f"Expected all constraints >= 0, got {constraints}"
        raise AssertionError(msg)
