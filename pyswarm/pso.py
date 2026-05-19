from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np


def _obj_wrapper(
    func: Callable, args: tuple, kwargs: dict, x: np.ndarray
) -> float:
    return func(x, *args, **kwargs)


def _is_feasible_wrapper(func: Callable, x: np.ndarray) -> bool:
    return np.all(func(x) >= 0)


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


def pso(  # noqa: PLR0913, PLR0917, PLR0912, PLR0915, PLR0914, PLR0911
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
) -> (
    tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray, np.ndarray]
):
    """
    Particle Swarm Optimizer (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array, optional
        The lower bounds of the design variable(s). Required if ``bounds``
        is not provided.
    ub : array, optional
        The upper bounds of the design variable(s). Required if ``bounds``
        is not provided.

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    seed : int, optional
        Seed for the random number generator used by particles and velocity
        updates. Use ``None`` for nondeterministic behavior (Default: None).
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.
    patience : int
        Maximum number of consecutive iterations without improvement before
        terminating early. Set to 0 to disable plateau detection (Default: 0).
        Useful for problems where the swarm reaches an exact optimum and no
        further improvement is possible (e.g. f(x)=0), preventing needless
        iterations until ``maxiter``.
    bounds : list of (lo, hi) pairs, optional
        SciPy-style bounds: a sequence of ``(lower, upper)`` tuples, one per
        design variable. Overrides ``lb`` and ``ub`` when provided. Example::

            bounds = [(-1, 1), (0, 5)]  # two variables
    intvar : list of int, optional
        Indices of design variables that must take integer values. After each
        position update the selected dimensions are rounded to the nearest
        integer and clipped to ``[lb, ub]``. Bounds for integer variables
        should themselves be integers (Default: None).
    init : array of shape (swarmsize, len(lb)), optional
        Custom initial particle positions. Each row is one particle. Values
        are clipped to ``[lb, ub]`` if any are out of range. When omitted,
        positions are sampled uniformly at random (Default: None).
    pool : object with a map() method, optional
        A custom parallel-evaluation pool (e.g. ``multiprocessing.Pool``,
        ``ipyparallel.Client[:].map``, ``dask.distributed.Client``).
        When provided, the ``processes`` argument is ignored and the pool's
        lifecycle is managed entirely by the caller. When omitted and
        ``processes > 1``, an internal ``multiprocessing.Pool`` is created and
        closed automatically (Default: None).

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    Raises
    ======
    TypeError
        If ``func`` is not callable.
    ValueError
        If bounds are mismatched or upper bounds are not all greater than lower.

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
    lb = np.array(lb)
    ub = np.array(ub)
    if not np.all(ub > lb):
        msg = "All upper-bound values must be greater than lower-bound values"
        raise ValueError(msg)

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    rng = np.random.default_rng(seed)

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)

    # Check for constraint function(s) #########################################
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

    # Set up the evaluation pool.
    owns_pool = False
    if pool is not None:
        mp_pool: Any = pool
    elif processes > 1:
        mp_pool = multiprocessing.Pool(processes)
        owns_pool = True
    else:
        mp_pool = None

    try:
        # Initialize the particle swarm ########################################
        swarm_size = swarmsize
        num_dims = len(lb)  # the number of dimensions each particle has
        x = rng.random((swarm_size, num_dims))  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(swarm_size)  # current particle function values
        fs = np.zeros(swarm_size, dtype=bool)  # feasibility of each particle
        fp = np.ones(swarm_size) * np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value

        # Initialize the particle's position
        if init is not None:
            x = np.asarray(init, dtype=float)
            if x.shape != (swarm_size, num_dims):
                msg = (
                    f"init must have shape ({swarm_size}, {num_dims}),"
                    f" got {x.shape}"
                )
                raise ValueError(msg)
            x = np.clip(x, lb, ub)
        else:
            x = lb + x * (ub - lb)
        if intvar is not None:
            x[:, intvar] = np.round(x[:, intvar]).clip(lb[intvar], ub[intvar])

        # Calculate objective and constraints for each particle
        if mp_pool is not None:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(swarm_size):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()

        # Initialize the particle's velocity
        v = rng.uniform(vlow, vhigh, size=(swarm_size, num_dims))

        # Iterate until termination criterion met ##############################
        it = 1
        stagnation = 0  # consecutive iterations without improvement
        while it <= maxiter:
            rp = rng.uniform(size=(swarm_size, num_dims))
            rg = rng.uniform(size=(swarm_size, num_dims))

            # Update the particles velocities
            v = omega * v + phip * rp * (p - x) + phig * rg * (g - x)
            # Update the particles' positions
            x += v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x * (~np.logical_or(maskl, masku)) + lb * maskl + ub * masku
            if intvar is not None:
                x[:, intvar] = np.round(x[:, intvar]).clip(
                    lb[intvar], ub[intvar]
                )

            # Update objectives and constraints
            if mp_pool is not None:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(swarm_size):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])

            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]

            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                stagnation = 0
                if debug:
                    print(
                        f"New best for swarm at iteration {it}: {p[i_min, :]} {fp[i_min]}"  # noqa: E501
                    )

                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min) ** 2))

                if np.abs(fg - fp[i_min]) <= minfunc:
                    if debug:
                        print(
                            f"Stopping search: Swarm best objective change less than {minfunc}"  # noqa: E501
                        )
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    return p_min, fp[i_min]
                if stepsize <= minstep:
                    if debug:
                        print(
                            f"Stopping search: Swarm best position change less than {minstep}"  # noqa: E501
                        )
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    return p_min, fp[i_min]
                g = p_min.copy()
                fg = fp[i_min]
            else:
                stagnation += 1
                if patience > 0 and stagnation >= patience:
                    if debug:
                        print(
                            f"Stopping search: no improvement for {patience} consecutive iterations"  # noqa: E501
                        )
                    if particle_output:
                        return g, fg, p, fp
                    return g, fg

            if debug:
                print(f"Best after iteration {it}: {g} {fg}")
            it += 1

        if debug:
            print(f"Stopping search: maximum iterations reached --> {maxiter}")

        if not is_feasible(g):
            print(
                "However, the optimization couldn't find a feasible design. Sorry"  # noqa: E501
            )
        if particle_output:
            return g, fg, p, fp
        return g, fg
    finally:
        if owns_pool and mp_pool is not None:
            mp_pool.terminate()
            mp_pool.join()
