# API Reference

All public symbols are exported from the `pyswarm` package.

## Exported symbols

- `pso`

---

## `pso`

### `pso(func, lb, ub, ...)`

Minimize an objective function using Particle Swarm Optimization.

```python
pso(
    func,
    lb,
    ub,
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
    processes=1,
    particle_output=False,
)
```

### Required arguments

`func`
: Callable objective function to minimize. It receives the current candidate
  vector as the first argument and returns a scalar objective value.

`lb`
: Lower bounds for each design variable.

`ub`
: Upper bounds for each design variable.

### Optional arguments

`ieqcons`
: List of constraint functions. Each function must return a value greater than
  or equal to `0.0` for a feasible candidate. If `f_ieqcons` is provided, this
  list is ignored.

`f_ieqcons`
: A single constraint function returning a 1-D array-like object. Every returned
  value must be greater than or equal to `0.0` for a feasible candidate.

`args`
: Additional positional arguments passed to the objective and constraint
  functions.

`kwargs`
: Additional keyword arguments passed to the objective and constraint functions.

`swarmsize`
: Number of particles in the swarm. Default: `100`.

`omega`
: Particle velocity scaling factor. Default: `0.5`.

`phip`
: Scaling factor for movement toward each particle's best known position.
  Default: `0.5`.

`phig`
: Scaling factor for movement toward the swarm's best known position.
  Default: `0.5`.

`maxiter`
: Maximum number of iterations. Default: `100`.

`minstep`
: Minimum step size of the swarm's best position before the search terminates.
  Default: `1e-8`.

`minfunc`
: Minimum change in the swarm's best objective value before the search
  terminates. Default: `1e-8`.

`debug`
: If `True`, prints progress information during optimization. Default: `False`.

`processes`
: Number of processes used to evaluate the objective and constraints. Default:
  `1`.

`particle_output`
: If `True`, returns per-particle best positions and objective values in
  addition to the swarm best. Default: `False`.

### Returns

By default, `pso` returns:

`xopt`
: The swarm's best known position.

`fopt`
: The objective value at `xopt`.

If `particle_output=True`, it also returns:

`p`
: The best known position per particle.

`pf`
: The objective values at each position in `p`.

### Raises

`TypeError`
: Raised when `func` is not callable.

`ValueError`
: Raised when lower and upper bounds have different lengths or when any upper
  bound is not greater than its corresponding lower bound.

## Constraint conventions

Constraints are feasible when they are non-negative:

```python
def constraint(x):
    return [limit - measured_value]
```

Use `f_ieqcons` when one function returns all constraint values. Use `ieqcons`
when each constraint is its own scalar function.

## References

See [References](references.md) for full citations.
