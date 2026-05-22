import numpy as np
from numpy.typing import NDArray

from pyswarm import pso


def banana_func(x: NDArray[np.floating]) -> np.floating:
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


def test_init_positions_respected() -> None:
    rng = np.random.default_rng(0)
    lb, ub = np.array([-3.0, -1.0]), np.array([2.0, 6.0])
    swarmsize = 20
    init = lb + rng.random((swarmsize, 2)) * (ub - lb)
    result = pso(banana_func, lb, ub, swarmsize=swarmsize, init=init, seed=0)
    xopt, fopt = result.x, result.fun
    if not np.allclose(xopt, [1.0, 1.0], atol=0.1):
        msg = f"Expected xopt near [1, 1], got {xopt}"
        raise AssertionError(msg)
    if not np.isclose(fopt, 4.0, atol=0.1):
        msg = f"Expected fopt near 4.0, got {fopt}"
        raise AssertionError(msg)


def test_init_wrong_shape_raises() -> None:
    bad_init = np.zeros((5, 3))
    try:
        pso(banana_func, [-1, -1], [1, 1], swarmsize=10, init=bad_init)
    except ValueError:
        pass
    else:
        msg = "Expected ValueError for wrong init shape"
        raise AssertionError(msg)
