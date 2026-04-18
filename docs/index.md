# PySwarm

**Particle Swarm Optimization (PSO) for Python**

`pyswarm` is a gradient-free, evolutionary optimization library for Python that implements [**Particle Swarm Optimization (PSO)**](https://en.wikipedia.org/wiki/Particle_swarm_optimization) with built-in support for constraints. It is lightweight, easy to use, and suitable for a wide range of optimization problems where gradient information is unavailable or impractical to compute.

## Features

- Gradient-free minimization for continuous design variables.
- Lower and upper bounds for every variable.
- Inequality constraints through either `ieqcons` or `f_ieqcons`.
- Optional multiprocessing for objective and constraint evaluation.
- Optional per-particle output for deeper inspection of a run.
- A small public API centered on one function: `pso`.

## Mathematical model

`pyswarm` solves minimization problems of the form

$$
\min_{\mathbf{x}} f(\mathbf{x})
$$

subject to bound constraints

$$
\mathbf{l_b} \le \mathbf{x} \le \mathbf{u_b}
$$

and optional inequality constraints

$$
c_j(\mathbf{x}) \ge 0.
$$

Each particle stores a current position, a velocity, its own best position, and
the swarm's best known position. The optimizer updates the swarm with the
standard PSO velocity rule:

$$
\mathbf{v} \leftarrow \omega\mathbf{v}
  + \phi_p r_p(\mathbf{p}-\mathbf{x})
  + \phi_g r_g(\mathbf{g}-\mathbf{x})
$$

$$
\mathbf{x} \leftarrow \mathbf{x} + \mathbf{v}.
$$

## Public API

- `pso` - minimize an objective function over bounded variables, optionally
  with inequality constraints.

## Documentation

- [Theory](theory.md) - mathematical background, hierarchical basis, algorithms
- [Installation](installation.md) - installation guide
- [Quickstart](quickstart.md) - runnable examples
- [API Reference](api.md) - function signature and arguments
- [References](references.md) - literature citations
