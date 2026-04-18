import numpy as np
from numpy.typing import NDArray

from pyswarm.pso import pso


def weight(x: NDArray[np.floating], *args: float) -> np.floating:
    h, d, t = x
    b, rho, _e, _p = args
    return rho * 2 * np.pi * d * t * np.sqrt((b / 2) ** 2 + h**2)


def stress(x: NDArray[np.floating], *args: float) -> np.floating:
    h, d, t = x
    b, _rho, _e, p = args
    return (p * np.sqrt((b / 2) ** 2 + h**2)) / (2 * t * np.pi * d * h)


def buckling_stress(x: NDArray[np.floating], *args: float) -> np.floating:
    h, d, t = x
    b, _rho, e, _p = args
    return (np.pi**2 * e * (d**2 + t**2)) / (8 * ((b / 2) ** 2 + h**2))


def deflection(x: NDArray[np.floating], *args: float) -> np.floating:
    h, d, t = x
    b, _rho, e, p = args
    return (p * np.sqrt((b / 2) ** 2 + h**2) ** 3) / (
        2 * t * np.pi * d * h**2 * e
    )


def truss_constraints(
    x: NDArray[np.floating], *args: float
) -> list[np.floating]:
    strs = stress(x, *args)
    buck = buckling_stress(x, *args)
    defl = deflection(x, *args)
    return [100 - strs, buck - strs, 0.25 - defl]


def test_twobar_truss() -> None:
    b = 60
    rho = 0.3
    e = 30000
    p = 66
    args = (b, rho, e, p)
    lb = [10, 1, 0.01]
    ub = [30, 3, 0.25]
    xopt, fopt = pso(
        weight,
        lb,
        ub,
        f_ieqcons=truss_constraints,
        args=args,
        maxiter=100,
        debug=True,
        seed=0,
    )
    if not np.isclose(fopt, 11.94, atol=0.1):
        msg = f"Expected fopt close to 11.94, got {fopt}"
        raise AssertionError(msg)
    if not all(c >= 0 for c in truss_constraints(xopt, *args)):
        msg = "Expected all constraints >= 0"
        raise AssertionError(msg)
