from collections.abc import Callable

import numpy as np
import pytest

from pyswarm import pso


def integer_func(x: np.ndarray) -> float:
    # Minimised at x=[2, 3] with f=0; both variables should be integers.
    return float((x[0] - 2) ** 2 + (x[1] - 3) ** 2)


def test_intvar_solutions_are_integers() -> None:
    lb = [0, 0]
    ub = [5, 5]
    result = pso(
        integer_func, lb, ub, intvar=[0, 1], swarmsize=30, seed=0, maxiter=200
    )
    xopt, fopt = result.x, result.fun
    if not np.allclose(xopt, np.round(xopt)):
        msg = f"Expected integer solution, got {xopt}"
        raise AssertionError(msg)
    if not np.isclose(fopt, 0.0, atol=0.1):
        msg = f"Expected fopt near 0.0, got {fopt}"
        raise AssertionError(msg)


@pytest.mark.parametrize("bad", [[-1], [2], [0, -2]])
def test_intvar_out_of_range_raises(bad: list[int]) -> None:
    with pytest.raises(ValueError, match="intvar indices"):
        pso(integer_func, [0, 0], [5, 5], intvar=bad, swarmsize=5)


@pytest.mark.parametrize("intvar", [[], [1]])
def test_intvar_boundary_indices_are_accepted(intvar: list[int]) -> None:
    # [] disables integer rounding; [1] is the last valid index (ndim - 1).
    result = pso(
        integer_func,
        [0, 0],
        [5, 5],
        intvar=intvar,
        swarmsize=5,
        maxiter=5,
        seed=0,
    )
    if result.x.shape != (2,):
        msg = f"Expected x of shape (2,), got {result.x.shape}"
        raise AssertionError(msg)
    if not np.allclose(result.x[intvar], np.round(result.x[intvar])):
        msg = f"Expected integer values at {intvar}, got {result.x}"
        raise AssertionError(msg)


@pytest.mark.parametrize("bad", [[0.5], [1.0], [True, False]])
def test_intvar_non_integer_raises(bad: list[object]) -> None:
    with pytest.raises(TypeError, match="intvar must contain integer"):
        pso(integer_func, [0, 0], [5, 5], intvar=bad, swarmsize=5)


@pytest.mark.parametrize("make", [iter, tuple, np.asarray])
def test_intvar_accepts_any_iterable(
    make: Callable[[list[int]], object],
) -> None:
    # iter() is a one-shot iterator: validation must not exhaust it.
    result = pso(
        integer_func,
        [0, 0],
        [5, 5],
        intvar=make([0, 1]),
        swarmsize=30,
        seed=0,
        maxiter=200,
    )
    if not np.allclose(result.x, np.round(result.x)):
        msg = f"Expected integer solution, got {result.x}"
        raise AssertionError(msg)
