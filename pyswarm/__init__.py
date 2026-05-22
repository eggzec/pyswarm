"""
===========================================
PySwarm: Particle Swarm Optimization (PSO)
===========================================

Fortran 77 back-end compiled via f2py / meson-python.

The complete PSO iteration kernel (personal-best update, global-best
update, velocity update, position update + clamping, integer rounding)
lives in a single Fortran 77 subroutine (``src/pso.f``) and is exposed
through the ``_pyswarm`` extension module.  The Python layer drives the
outer optimisation loop and evaluates the (arbitrary Python-callable)
objective function.

Authors: Abraham Lee
         Sebastian Castillo-Hair
         Saud Zahir

Copyright (c) 2013, Abraham D. Lee
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from pyswarm._pyswarm import pso as _pso_step  # noqa: PLC0414


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obj_wrapper(
    func: Callable, args: tuple, kwargs: dict, x: np.ndarray
) -> float:
    return func(x, *args, **kwargs)


def _is_feasible_wrapper(func: Callable, x: np.ndarray) -> bool:
    return bool(np.all(func(x) >= 0))


def _cons_none_wrapper(x: np.ndarray) -> np.ndarray:
    return np.array([0])


def _cons_ieqcons_wrapper(
    ieqcons: list, args: tuple, kwargs: dict, x: np.ndarray
) -> np.ndarray:
    return np.array([y(x, *args, **kwargs) for y in ieqcons])


def _cons_f_ieqcons_wrapper(
    f_ieqcons: Callable, args: tuple, kwargs: dict, x: np.ndarray
) -> np.ndarray:
    return np.array(f_ieqcons(x, *args, **kwargs))


# ---------------------------------------------------------------------------
# OptimizeResult  (scipy.optimize-compatible)
# ---------------------------------------------------------------------------

class OptimizeResult(dict):  # type: ignore[type-arg]
    """Optimisation result returned by :func:`pso`.

    Behaves like a :class:`dict` but attributes are accessible with dot
    notation, matching the interface of ``scipy.optimize.OptimizeResult``.

    Attributes
    ----------
    x : ndarray
        Best position found (global optimum estimate).
    fun : float
        Objective value at *x*.
    success : bool
        ``True`` when a feasible solution was found.
    message : str
        Human-readable reason for termination.
    nit : int
        Number of PSO iterations performed.
    nfev : int
        Total number of objective-function evaluations.
    particles : ndarray, optional
        Best per-particle positions (present only when *particle_output* is
        ``True``).
    particle_fun : ndarray, optional
        Objective values at each position in *particles* (present only when
        *particle_output* is ``True``).
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pso(  # noqa: PLR0913, PLR0917, PLR0912, PLR0914, PLR0911
    func: Callable,
    lb: list | np.ndarray | None = None,
    ub: list | np.ndarray | None = None,
    ieqcons: list | None = None,
    f_ieqcons: Callable | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    swarmsize: int = 100,
    omega: float = 0.5,
    phip: float = 0.5,
    phig: float = 0.5,
    maxiter: int = 100,
    minstep: float = 1e-8,
    minfunc: float = 1e-8,
    debug: bool = False,  # noqa: FBT001, FBT002
    processes: int = 1,
    particle_output: bool = False,  # noqa: FBT001, FBT002
    seed: int | None = None,
    patience: int = 0,
    bounds: list[tuple[float, float]] | None = None,
    pool: object | None = None,
    init: np.ndarray | None = None,
    intvar: list[int] | None = None,
) -> OptimizeResult:
    """Minimise *func* with the Particle Swarm Optimisation algorithm.

    The numerical PSO kernel runs entirely inside a single compiled
    Fortran 77 subroutine (see ``src/pso.f``).  Python drives the outer
    loop, evaluates the objective and constraint functions, and collects
    results in a :class:`OptimizeResult`.

    Parameters
    ----------
    func : callable
        Objective function to minimise.  Called as
        ``func(x, *args, **kwargs)``; must return a scalar.
    lb : array-like, optional
        Lower bounds of the design variables.  Required when *bounds* is
        not supplied.
    ub : array-like, optional
        Upper bounds of the design variables.  Required when *bounds* is
        not supplied.
    ieqcons : list of callables, optional
        Inequality constraints of the form ``c_j(x, *args) >= 0``.
    f_ieqcons : callable, optional
        Vectorised constraint: returns a 1-D array whose every element
        must be ``>= 0``.  Overrides *ieqcons* when provided.
    args : tuple, optional
        Extra positional arguments forwarded to *func* and constraints.
    kwargs : dict, optional
        Extra keyword arguments forwarded to *func* and constraints.
    swarmsize : int, optional
        Number of particles (default 100).
    omega : float, optional
        Inertia weight (default 0.5).
    phip : float, optional
        Cognitive (personal-best) coefficient (default 0.5).
    phig : float, optional
        Social (global-best) coefficient (default 0.5).
    maxiter : int, optional
        Maximum number of iterations (default 100).
    minstep : float, optional
        Convergence threshold on the Euclidean step size of the global
        best (default 1e-8).
    minfunc : float, optional
        Convergence threshold on the absolute objective improvement
        (default 1e-8).
    debug : bool, optional
        Print iteration-level diagnostics (default ``False``).
    processes : int, optional
        Worker processes for objective evaluation (default 1).
    particle_output : bool, optional
        Include per-particle best positions and values in the result
        (default ``False``).
    seed : int or None, optional
        RNG seed for reproducibility (default ``None``).
    patience : int, optional
        Stop after this many consecutive non-improving iterations;
        ``0`` disables early stopping (default 0).
    bounds : list of (lo, hi) pairs, optional
        SciPy-style bounds; overrides *lb* / *ub* when provided.
    pool : object with ``map()``, optional
        External parallel pool; *processes* is ignored when set.
    init : ndarray of shape (swarmsize, ndim), optional
        Custom initial particle positions; clipped to ``[lb, ub]``.
    intvar : list of int, optional
        0-based indices of design variables constrained to integers.

    Returns
    -------
    result : OptimizeResult
        Dict-like result compatible with ``scipy.optimize.OptimizeResult``
        with the following fields:

        x : ndarray
            Best position found.
        fun : float
            Objective value at *x*.
        success : bool
            ``True`` when a feasible solution was found.
        message : str
            Reason for termination.
        nit : int
            Number of iterations performed.
        nfev : int
            Total objective-function evaluations.
        particles : ndarray
            Best per-particle positions (*particle_output* only).
        particle_fun : ndarray
            Objective values at *particles* (*particle_output* only).

    Raises
    ------
    TypeError
        If *func* is not callable.
    ValueError
        If bounds are mismatched or any upper bound is not strictly
        greater than the corresponding lower bound.
    """
    if kwargs is None:
        kwargs = {}
    if ieqcons is None:
        ieqcons = []
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

    vhigh: np.ndarray = np.abs(ub - lb)
    vlow: np.ndarray = -vhigh

    rng = np.random.default_rng(seed)

    # --- wrap objective and constraints ------------------------------------
    obj = partial(_obj_wrapper, func, args, kwargs)

    if f_ieqcons is None:
        if not ieqcons:
            if debug:
                print("No constraints given.")
            cons = _cons_none_wrapper
        else:
            if debug:
                print("Converting ieqcons to a single constraint function")
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print("Single constraint function given in f_ieqcons")
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # --- pool setup -------------------------------------------------------
    owns_pool = False
    mp_pool: Any = None
    if pool is not None:
        mp_pool = pool
    elif processes > 1:
        mp_pool = multiprocessing.Pool(processes)
        owns_pool = True

    # --- integer variable indices (convert to 1-based Fortran) -----------
    # Fortran ivar must be at least length-1 even when NIVAR=0 so the
    # array always has a valid address; the subroutine guards with NIVAR>0.
    if intvar is not None and len(intvar) > 0:
        ivar_f = np.asarray([i + 1 for i in intvar], dtype=np.int32)
    else:
        ivar_f = np.zeros(1, dtype=np.int32)  # placeholder, NIVAR=0 in Fortran

    # When NIVAR==0 we pass a dummy 1-element ivar so f2py len(ivar)==1,
    # but the subroutine skips step 5 when NIVAR==0.
    # We override nivar explicitly to 0 via the pyf hide rule that reads
    # len(ivar) -- so we need ivar to actually be length 0 when no int vars.
    # Use a proper empty array when there are no integer variables.
    if intvar is None or len(intvar) == 0:
        ivar_f = np.empty(0, dtype=np.int32)

    try:
        S = swarmsize
        D = len(lb)
        nfev = 0

        # ------------------------------------------------------------------
        # Initialise positions
        # ------------------------------------------------------------------
        if init is not None:
            x = np.asfortranarray(
                np.clip(np.asarray(init, dtype=np.float64), lb, ub)
            )
            if x.shape != (S, D):
                msg = f"init must have shape ({S}, {D}), got {x.shape}"
                raise ValueError(msg)
        else:
            x = np.asfortranarray(
                lb + rng.random((S, D)) * (ub - lb)
            )

        # Apply integer rounding to initial positions
        if intvar is not None and len(intvar) > 0:
            x[:, intvar] = np.round(x[:, intvar]).clip(
                lb[intvar], ub[intvar]
            )

        # ------------------------------------------------------------------
        # Initialise velocities
        # ------------------------------------------------------------------
        v = np.asfortranarray(
            rng.uniform(vlow, vhigh, size=(S, D))
        )

        # ------------------------------------------------------------------
        # Initialise personal and global bests
        # ------------------------------------------------------------------
        p  = np.asfortranarray(np.zeros((S, D), dtype=np.float64))
        fp = np.ones(S, dtype=np.float64) * np.inf
        fg = np.inf
        g  = np.asfortranarray(x[0, :].copy())  # fallback

        # ------------------------------------------------------------------
        # Evaluate initial positions
        # ------------------------------------------------------------------
        fx = np.zeros(S, dtype=np.float64)
        fs = np.zeros(S, dtype=np.int32)

        if mp_pool is not None:
            fx[:] = np.array(mp_pool.map(obj, x))
            fs[:] = np.array(mp_pool.map(is_feasible, x), dtype=np.int32)
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = int(is_feasible(x[i, :]))
        nfev += S

        # Bootstrap personal and global bests from initial evaluation
        for i in range(S):
            if fs[i] and fx[i] < fp[i]:
                p[i, :] = x[i, :]
                fp[i] = fx[i]

        i_min = int(np.argmin(fp))
        if fp[i_min] < fg:
            fg = fp[i_min]
            g  = np.asfortranarray(p[i_min, :].copy())

        # ------------------------------------------------------------------
        # Main optimisation loop
        # ------------------------------------------------------------------
        it = 1
        stagnation = 0
        message = f"Maximum iterations reached: {maxiter}"

        while it <= maxiter:
            rp    = np.asfortranarray(rng.random((S, D)))
            rg_rnd = np.asfortranarray(rng.random((S, D)))

            # --- ONE complete PSO step (Fortran 77 kernel) ----------------
            (
                x, v, p, fp, g, fg,
                stepsize, funcdiff, improved,
            ) = _pso_step(
                x, v, p, fp, g, fg, fx, fs,
                rp, rg_rnd, lb, ub, ivar_f,
                float(omega), float(phip), float(phig),
                float(minstep), float(minfunc),
            )

            # --- Evaluate objective at new positions ----------------------
            if mp_pool is not None:
                fx[:] = np.array(mp_pool.map(obj, x))
                fs[:] = np.array(
                    mp_pool.map(is_feasible, x), dtype=np.int32
                )
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = int(is_feasible(x[i, :]))
            nfev += S

            # --- Convergence checks ---------------------------------------
            if improved:
                stagnation = 0
                if debug:
                    print(
                        f"New best for swarm at iteration {it}: "
                        f"{np.asarray(g)} {float(fg)}"
                    )
                if funcdiff <= minfunc:
                    if debug:
                        print(
                            "Stopping search: Swarm best objective change"
                            f" less than {minfunc}"
                        )
                    message = (
                        f"Objective improvement below minfunc={minfunc}"
                    )
                    break
                if stepsize <= minstep:
                    if debug:
                        print(
                            "Stopping search: Swarm best position change"
                            f" less than {minstep}"
                        )
                    message = f"Step size below minstep={minstep}"
                    break
            else:
                stagnation += 1
                if patience > 0 and stagnation >= patience:
                    if debug:
                        print(
                            "Stopping search: no improvement for "
                            f"{patience} consecutive iterations"
                        )
                    message = (
                        f"No improvement for {patience} consecutive "
                        "iterations (patience)"
                    )
                    break

            if debug:
                print(f"Best after iteration {it}: {np.asarray(g)} {float(fg)}")
            it += 1

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        success = is_feasible(np.asarray(g))
        if not success:
            print(
                "However, the optimization couldn't find a feasible"
                " design. Sorry"
            )

        result = OptimizeResult(
            x=np.asarray(g, dtype=np.float64),
            fun=float(fg),
            success=bool(success),
            message=message,
            nit=it - 1,
            nfev=nfev,
        )
        if particle_output:
            result.particles    = np.asarray(p, dtype=np.float64)
            result.particle_fun = np.asarray(fp, dtype=np.float64)
        return result

    finally:
        if owns_pool and mp_pool is not None:
            mp_pool.terminate()
            mp_pool.join()


__all__ = ["OptimizeResult", "pso"]
