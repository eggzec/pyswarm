"""
===========================================
PySwarm: Particle Swarm Optimization (PSO)
===========================================

A gradient-free, evolutionary optimization
library for Python that implements Particle
Swarm Optimization (PSO) with built-in
support for constraints. It is lightweight,
easy to use, and suitable for a wide range
of optimization problems where gradient
information is unavailable or impractical
to compute.

Authors: Abraham Lee
         Sebastian Castillo-Hair
         Saud Zahir

Copyright (c) 2013, Abraham D. Lee
Copyright (c) 2026, eggzec
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from pyswarm._pyswarm import pso as _pso


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"


# -------------------------------
# Constraint / objective helpers
# -------------------------------


def _obj_wrapper(
    func: Callable[..., float],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    x: np.ndarray,
) -> float:
    return func(x, *args, **kwargs)


def _is_feasible_wrapper(
    func: Callable[..., np.ndarray], x: np.ndarray
) -> bool:
    return bool(np.all(func(x) >= 0))


def _cons_none_wrapper(x: np.ndarray) -> np.ndarray:
    return np.array([0.0])


def _cons_ieqcons_wrapper(
    ieqcons: list[Callable[..., float]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    x: np.ndarray,
) -> np.ndarray:
    return np.array([c(x, *args, **kwargs) for c in ieqcons])


def _cons_f_ieqcons_wrapper(
    f_ieqcons: Callable[..., np.ndarray],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    x: np.ndarray,
) -> np.ndarray:
    return np.array(f_ieqcons(x, *args, **kwargs))


class OptimizeResult(dict):  # type: ignore[type-arg]
    """Optimisation result returned by :func:`pso`.

    Behaves like a :class:`dict` with attribute access, compatible with
    ``scipy.optimize.OptimizeResult``.

    Attributes
    ----------
    x : ndarray
        Best position found.
    fun : float
        Objective value at ``x``.
    success : bool
        ``True`` when a feasible solution was found.
    message : str
        Human-readable reason for termination.
    nit : int
        Number of PSO iterations performed.
    nfev : int
        Total number of objective-function evaluations.
    particles : ndarray, optional
        Best per-particle positions (only when ``particle_output=True``).
    particle_fun : ndarray, optional
        Objective values at ``particles`` (only when ``particle_output=True``).
    """

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.items()))
        return f"{self.__class__.__name__}({items})"


# ------------
# Status code
# ------------
_STATUS_MSG: dict[int, str] = {
    0: "Maximum iterations reached",
    1: "Step size below minstep threshold",
    2: "Objective improvement below minfunc threshold",
    3: "Early stop: no improvement for patience consecutive iterations",
}


# -----------
# Public API
# -----------


def pso(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
    func: Callable[..., float],
    lb: list[float] | np.ndarray | None = None,
    ub: list[float] | np.ndarray | None = None,
    ieqcons: list[Callable[..., float]] | None = None,
    f_ieqcons: Callable[..., np.ndarray] | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    swarmsize: int = 100,
    omega: float = 0.5,
    phip: float = 0.5,
    phig: float = 0.5,
    maxiter: int = 100,
    minstep: float = 1e-8,
    minfunc: float = 1e-8,
    debug: bool = False,  # noqa: FBT001, FBT002
    particle_output: bool = False,  # noqa: FBT001, FBT002
    seed: int | None = None,
    patience: int = 0,
    bounds: list[tuple[float, float]] | None = None,
    init: np.ndarray | None = None,
    intvar: list[int] | None = None,
) -> OptimizeResult:
    """Particle Swarm Optimizer (PSO)

    Parameters
    ----------
    func : callable
        Objective function: ``func(x, *args, **kwargs) -> float``.
    lb : array-like, optional
        Lower bounds.  Required when *bounds* is not supplied.
    ub : array-like, optional
        Upper bounds.  Required when *bounds* is not supplied.
    ieqcons : list of callables, optional
        Inequality constraints ``c_j(x, *args) >= 0``.
    f_ieqcons : callable, optional
        Vectorised constraint returning a 1-D array (each element >= 0).
        Overrides *ieqcons*.
    args : tuple, optional
        Extra positional arguments forwarded to *func* and constraints.
    kwargs : dict, optional
        Extra keyword arguments forwarded to *func* and constraints.
    swarmsize : int, optional
        Number of particles (default 100).
    omega : float, optional
        Inertia weight (default 0.5).
    phip : float, optional
        Cognitive coefficient (default 0.5).
    phig : float, optional
        Social coefficient (default 0.5).
    maxiter : int, optional
        Maximum iterations (default 100).
    minstep : float, optional
        Convergence threshold on the Euclidean step of the global best
        (default 1e-8).
    minfunc : float, optional
        Convergence threshold on the absolute objective improvement
        (default 1e-8).
    debug : bool, optional
        Print progress messages (default ``False``).
    particle_output : bool, optional
        Include per-particle bests in the result (default ``False``).
    seed : int or None, optional
        RNG seed for reproducibility (default ``None``).
    patience : int, optional
        Stop after this many consecutive non-improving iterations;
        ``0`` disables (default 0).
    bounds : list of (lo, hi) pairs, optional
        SciPy-style bounds; overrides *lb* / *ub*.
    init : ndarray of shape (swarmsize, ndim), optional
        Custom initial particle positions; clipped to ``[lb, ub]``.
    intvar : list of int, optional
        0-based indices of design variables constrained to integer values.

    Returns
    -------
    result : OptimizeResult
        scipy.optimize-compatible result with fields:
        ``x``, ``fun``, ``success``, ``message``, ``nit``, ``nfev``,
        and optionally ``particles`` / ``particle_fun``.

    Raises
    ------
    TypeError
        If *func* is not callable.
    ValueError
        If bounds are mismatched or any upper bound is not strictly greater
        than the corresponding lower bound.
    """
    if kwargs is None:
        kwargs = {}
    if ieqcons is None:
        ieqcons = []

    # --- Resolve bounds ---
    if bounds is not None:
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
    if lb is None or ub is None:
        msg = "Either 'bounds' or both 'lb' and 'ub' must be provided"
        raise ValueError(msg)
    if len(lb) != len(ub):
        msg = "Lower- and upper-bounds must be the same length"
        raise ValueError(msg)
    if not callable(func):
        msg = "Invalid function handle"
        raise TypeError(msg)

    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    if not np.all(ub > lb):
        msg = "All upper-bound values must be greater than lower-bound values"
        raise ValueError(msg)

    S: int = swarmsize  # noqa N806
    D: int = len(lb)  # noqa N806

    # --- Build objective + feasibility wrappers ---
    obj = partial(_obj_wrapper, func, args, kwargs)

    if f_ieqcons is None:
        cons = (
            _cons_none_wrapper
            if not ieqcons
            else partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        )
    else:
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Returns < 1e300  for feasible particles (the actual objective value),
    # Returns >= 1e300 for infeasible particles (infeasibility sentinel).
    INFEAS: float = 1.0e300  # noqa N806

    def _eval(x: np.ndarray, n: int) -> float:
        xarr = np.asarray(x, dtype=np.float64)
        if not is_feasible(xarr):
            return INFEAS
        return float(obj(xarr))

    # --- RNG + initial state ---
    rng = np.random.default_rng(seed)
    vhigh: np.ndarray = np.abs(ub - lb)
    vlow: np.ndarray = -vhigh

    if init is not None:
        x0 = np.asfortranarray(
            np.clip(np.asarray(init, dtype=np.float64), lb, ub)
        )
        if x0.shape != (S, D):
            msg = f"init must have shape ({S}, {D}), got {x0.shape}"
            raise ValueError(msg)
    else:
        x0 = np.asfortranarray(lb + rng.random((S, D)) * (ub - lb))

    # Apply integer rounding to initial positions in Python
    if intvar is not None and len(intvar) > 0:
        x0[:, intvar] = np.round(x0[:, intvar]).clip(lb[intvar], ub[intvar])

    # Initial velocities
    v0 = np.asfortranarray(rng.uniform(vlow, vhigh, size=(S, D)))

    rnd_p = np.asfortranarray(rng.random((maxiter, S, D)))
    rnd_g = np.asfortranarray(rng.random((maxiter, S, D)))

    if intvar is not None and len(intvar) > 0:
        ivar_f = np.asarray([i + 1 for i in intvar], dtype=np.int32)
    else:
        ivar_f = np.empty(0, dtype=np.int32)

    _, _, g, fg, p_out, fp_out, niters, status = _pso(
        _eval,
        lb,
        ub,
        x0,
        v0,
        rnd_p,
        rnd_g,
        float(omega),
        float(phip),
        float(phig),
        float(minstep),
        float(minfunc),
        int(patience),
        int(debug),
        ivar_f,
    )

    # nfev: S initial evals + S evals per iteration
    nfev: int = S + niters * S

    success = bool(fg < INFEAS) and bool(is_feasible(np.asarray(g)))
    if not success:
        print(
            "However, the optimization couldn't find a feasible design. Sorry"
        )

    result = OptimizeResult(
        x=np.asarray(g, dtype=np.float64),
        fun=float(fg) if fg < INFEAS else float("inf"),
        success=success,
        message=_STATUS_MSG.get(int(status), f"Unknown status {status}"),
        nit=int(niters),
        nfev=nfev,
    )
    if particle_output:
        result.particles = np.asarray(p_out, dtype=np.float64)
        result.particle_fun = np.asarray(fp_out, dtype=np.float64)
    return result


__all__ = ["OptimizeResult", "pso"]
