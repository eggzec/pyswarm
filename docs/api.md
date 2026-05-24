# API Reference

All public symbols are exported from the `pyswarm` package.

## Exported symbols

- [`pso`](#pso)
- [`OptimizeResult`](#optimizeresult)

---

## `pso`

### `pso(func, lb, ub, ...)`

Minimize an objective function using Particle Swarm Optimization.

```python
pso(
    func,
    lb=None,
    ub=None,
    ieqcons=None,
    f_ieqcons=None,
    args=(),
    kwargs=None,
    swarmsize=100,
    omega=0.5,
    phip=0.5,
    phig=0.5,
    maxiter=100,
    minstep=1e-8,
    minfunc=1e-8,
    debug=False,
    particle_output=False,
    seed=None,
    patience=0,
    bounds=None,
    init=None,
    intvar=None,
)
```

### Required arguments

`func`
: Callable objective function to minimize. It receives the current candidate
  vector as the first argument and returns a scalar objective value.

`lb`
: Lower bounds for each design variable. Required when `bounds` is not
  provided.

`ub`
: Upper bounds for each design variable. Required when `bounds` is not
  provided.

### Optional arguments

`ieqcons`
: List of constraint functions. Each function must return a value greater than
  or equal to `0.0` for a feasible candidate. Ignored when `f_ieqcons` is
  provided.

`f_ieqcons`
: A single constraint function returning a 1-D array-like object. Every
  returned value must be greater than or equal to `0.0` for a feasible
  candidate. Overrides `ieqcons` when provided.

`args`
: Additional positional arguments forwarded to the objective and constraint
  functions.

`kwargs`
: Additional keyword arguments forwarded to the objective and constraint
  functions.

`swarmsize`
: Number of particles in the swarm. Default: `100`.

`omega`
: Inertia weight — scales the previous velocity. Default: `0.5`.

`phip`
: Cognitive coefficient — scales movement toward each particle's personal
  best. Default: `0.5`.

`phig`
: Social coefficient — scales movement toward the swarm's global best.
  Default: `0.5`.

`maxiter`
: Maximum number of iterations. Default: `100`.

`minstep`
: Convergence threshold on the Euclidean distance the global best moves
  between improvements. The search terminates when this threshold is met.
  Default: `1e-8`.

`minfunc`
: Convergence threshold on the absolute improvement in the global best
  objective value. The search terminates when this threshold is met.
  Default: `1e-8`.

`debug`
: If `True`, prints progress messages from the Fortran kernel to stdout.
  Default: `False`.

`particle_output`
: If `True`, the returned `OptimizeResult` includes `particles` (best
  per-particle positions) and `particle_fun` (objective values at those
  positions). Default: `False`.

`seed`
: Integer seed for the random number generator. Pass an integer for fully
  reproducible runs; `None` gives non-deterministic behavior. Default: `None`.

`patience`
: Maximum number of consecutive iterations without an improvement to the
  global best before the search terminates early. Set to `0` to disable
  patience-based early stopping. Default: `0`.

`bounds`
: SciPy-style sequence of `(lower, upper)` pairs, one per design variable.
  Overrides `lb` and `ub` when provided. Example: `bounds=[(-1, 1), (0, 5)]`.

`init`
: Custom initial particle positions as an array of shape
  `(swarmsize, ndim)`. Values outside `[lb, ub]` are clipped. When omitted,
  positions are sampled uniformly at random. Default: `None`.

`intvar`
: List of **0-based** indices of design variables that must take integer
  values. After each position update the selected dimensions are rounded to
  the nearest integer and clipped to `[lb, ub]`. Bounds for integer variables
  should themselves be integers. Default: `None`.

### Returns

`result` — an [`OptimizeResult`](#optimizeresult) with the following fields:

| Field | Type | Description |
|---|---|---|
| `x` | `ndarray` | Best position found (global optimum estimate) |
| `fun` | `float` | Objective value at `x` |
| `success` | `bool` | `True` when a feasible solution was found |
| `message` | `str` | Human-readable reason for termination |
| `nit` | `int` | Number of PSO iterations performed |
| `nfev` | `int` | Total number of objective-function evaluations |
| `particles` | `ndarray` | Best per-particle positions (`particle_output=True` only) |
| `particle_fun` | `ndarray` | Objective values at `particles` (`particle_output=True` only) |

### Raises

`TypeError`
: Raised when `func` is not callable.

`ValueError`
: Raised when lower and upper bounds have different lengths, or when any
  upper bound is not strictly greater than its corresponding lower bound.

---

## `OptimizeResult`

A `dict` subclass with attribute access, compatible with
`scipy.optimize.OptimizeResult`.

```python
result = pso(func, lb, ub)

# Attribute access
print(result.x)  # best position
print(result.fun)  # objective value at result.x
print(result.success)  # True if a feasible solution was found
print(result.message)  # termination reason
print(result.nit)  # iterations performed
print(result.nfev)  # objective-function evaluations

# Dict-style access — equivalent to the above
print(result["x"])
print(result["fun"])
```

---

## Constraint conventions

Constraints are feasible when they are non-negative:

```python
def constraint(x):
    return [limit - measured_value]
```

Use `f_ieqcons` when one function returns all constraint values at once:

```python
def all_constraints(x):
    return [c1(x), c2(x), c3(x)]


result = pso(func, lb, ub, f_ieqcons=all_constraints)
```

Use `ieqcons` when each constraint is its own scalar function:

```python
result = pso(func, lb, ub, ieqcons=[c1, c2, c3])
```

---

## References

See [References](references.md) for full citations.
