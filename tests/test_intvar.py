import numpy as np

from pyswarm.pso import pso


def integer_func(x: np.ndarray) -> float:
    # Minimised at x=[2, 3] with f=0; both variables should be integers.
    return float((x[0] - 2) ** 2 + (x[1] - 3) ** 2)


def test_intvar_solutions_are_integers() -> None:
    lb = [0, 0]
    ub = [5, 5]
    xopt, fopt = pso(
        integer_func, lb, ub, intvar=[0, 1], swarmsize=30, seed=0, maxiter=200
    )
    if not np.allclose(xopt, np.round(xopt)):
        msg = f"Expected integer solution, got {xopt}"
        raise AssertionError(msg)
    if not np.isclose(fopt, 0.0, atol=0.1):
        msg = f"Expected fopt near 0.0, got {fopt}"
        raise AssertionError(msg)
