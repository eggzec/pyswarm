# Quickstart

The examples below cover unconstrained optimization, constrained optimization, and a small
engineering design problem.

## Example 1: banana function without constraints

Minimize the fourth-order banana function over two bounded variables:

```python
from pyswarm import pso


def banana_func(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


lb = [-3, -1]
ub = [2, 6]

xopt, fopt = pso(banana_func, lb, ub)
print(f"xopt: {xopt}")
print(f"fopt: {fopt}")
```

Expected result:

- `xopt = [1.0, 1.0]`
- `fopt = 4.0`

## Example 2: banana function with a constraint

Add an inequality constraint. A candidate is feasible when every returned
constraint value is greater than or equal to `0.0`:

```python
from pyswarm import pso


def banana_func(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


def banana_constraint(x):
    x1 = x[0]
    x2 = x[1]
    return [-((x1 + 0.25) ** 2) + 0.75 * x2]


lb = [-3, -1]
ub = [2, 6]

xopt, fopt = pso(banana_func, lb, ub, f_ieqcons=banana_constraint)
print(f"xopt: {xopt}")
print(f"fopt: {fopt}")
print(f"constraint: {banana_constraint(xopt)}")
```

Expected result:

- `xopt` close to `[0.5, 0.75]`
- `fopt` close to `4.5`
- every value returned by `banana_constraint(xopt)` to be non-negative

## Example 3: two-bar truss optimization

Another useful example is in the design of a two-bar truss in the shape of an
A-frame. The objective of the problem is to minimize the weight of the truss
while satisfying three design constraints:

- Yield Stress <= 100 kpsi
- Yield Stress <= Buckling Stress
- Deflection <= 0.25 in

The design variables are:

- `H`: the height of the truss, in inches
- `d`: the diameter of the truss tubes, in inches
- `t`: the wall thickness of the tubes, in inches

Other parameters that will be held constant are:

- `B`: the base separation distance, in inches
- `rho`: the density of the truss material, in lb/in^3
- `E`: the modulus of elasticity of the truss material, in kpsi (1000-psi)
- `P`: the downward vertical load on the top of the truss, in kip (1000-lbf)

This example shows how the optional `args` parameter may be used to pass other
needed values to the objective and constraint functions.

```python
import numpy as np
from pyswarm import pso


def weight(x, *args):
    h, d, t = x
    b, rho, _e, _p = args
    return rho * 2 * np.pi * d * t * np.sqrt((b / 2) ** 2 + h**2)


def stress(x, *args):
    h, d, t = x
    b, _rho, _e, p = args
    return (p * np.sqrt((b / 2) ** 2 + h**2)) / (2 * t * np.pi * d * h)


def buckling_stress(x, *args):
    h, d, t = x
    b, _rho, e, _p = args
    return (np.pi**2 * e * (d**2 + t**2)) / (8 * ((b / 2) ** 2 + h**2))


def deflection(x, *args):
    h, d, t = x
    b, _rho, e, p = args
    return (p * np.sqrt((b / 2) ** 2 + h**2) ** 3) / (
        2 * t * np.pi * d * h**2 * e
    )


def truss_constraints(x, *args):
    strs = stress(x, *args)
    buck = buckling_stress(x, *args)
    defl = deflection(x, *args)
    return [100 - strs, buck - strs, 0.25 - defl]


b = 60
rho = 0.3  # lb/in^3
e = 30000  # kpsi
p = 66
args = (b, rho, e, p)

lb = [10, 1, 0.01]
ub = [30, 3, 0.25]

xopt, fopt = pso(
    weight, lb, ub, f_ieqcons=truss_constraints, args=args, maxiter=100
)
print(f"xopt: {xopt}")
print(f"weight: {fopt}")
print(f"constraints: {truss_constraints(xopt, *args)}")
```

Expected result:

- `fopt` close to `11.94`
- every value returned by `truss_constraints(xopt, *args)` is non-negative

Because the algorithm is randomized, results can vary slightly between runs.
