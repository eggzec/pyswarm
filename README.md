# PySwarm

**Particle Swarm Optimization (PSO) for Python**

[![Tests](https://github.com/eggzec/pyswarm/actions/workflows/code_test.yml/badge.svg)](https://github.com/eggzec/pyswarm/actions/workflows/code_test.yml)
[![Documentation](https://github.com/eggzec/pyswarm/actions/workflows/docs_build.yml/badge.svg)](https://github.com/eggzec/pyswarm/actions/workflows/docs_build.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![codecov](https://codecov.io/github/eggzec/pyswarm/graph/badge.svg)](https://codecov.io/github/eggzec/pyswarm)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=eggzec_pyswarm&metric=alert_status)](https://sonarcloud.io/project/overview?id=eggzec_pyswarm)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](LICENSE)

[![PyPI Downloads](https://img.shields.io/pypi/dm/pyswarm.svg?label=PyPI%20downloads)](https://pypi.org/project/pyswarm/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyswarm.svg)](https://pypi.org/project/pyswarm/)

`pyswarm` is a gradient-free, evolutionary optimization library for Python that implements [**Particle Swarm Optimization (PSO)**](https://en.wikipedia.org/wiki/Particle_swarm_optimization) with built-in support for constraints. It is lightweight, easy to use, and suitable for a wide range of optimization problems where gradient information is unavailable or impractical to compute.

## Quick example

```python
from pyswarm import pso


def objective(x):
    x1, x2 = x
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5


lb = [-3, -1]
ub = [2, 6]

result = pso(objective, lb, ub)

print("Optimal solution:", result.x)  # → [1.0, 1.0]
print("Function value:", result.fun)  # → 4.0
print("Converged:", result.success)
print("Reason:", result.message)
print("Iterations:", result.nit)
print("Evaluations:", result.nfev)
```

`pso` returns an `OptimizeResult` — a `dict` subclass with attribute
access, compatible with `scipy.optimize.OptimizeResult`.

## Installation

```bash
pip install pyswarm
```

Requires Python 3.10+ and NumPy. See the
[full installation guide](https://eggzec.github.io/pyswarm/installation/) for
uv, poetry, and source builds.

## Documentation

- [Theory](https://eggzec.github.io/pyswarm/theory/) — mathematical background, hierarchical basis, algorithms
- [Quickstart](https://eggzec.github.io/pyswarm/quickstart/) — runnable examples
- [API Reference](https://eggzec.github.io/pyswarm/api/) — class and function signatures
- [References](https://eggzec.github.io/pyswarm/references/) — literature citations

## License

BSD-3-Clause — see [LICENSE](LICENSE).
