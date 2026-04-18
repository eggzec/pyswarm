# Theory

## 1) Particle swarm optimization

Particle Swarm Optimization (PSO) is a population-based optimization method. It
iteratively improves a population of candidate solutions, called particles,
with respect to an objective function.

PSO is a metaheuristic. It makes few assumptions about the problem, can search
large spaces of candidate solutions, and does not require gradients. That makes
it useful for nonlinear, noisy, or derivative-free optimization problems. Like
other metaheuristics, PSO does not guarantee that a global optimum will be
found.

## 2) Particles and the objective function

For a minimization problem, let

$$
f: \mathbb{R}^{n} \rightarrow \mathbb{R}
$$

be the objective function. A candidate solution is a vector

$$
\mathbf{x}_i \in \mathbb{R}^{n}.
$$

Each particle has:

- a position $\mathbf{x}_i$
- a velocity $\mathbf{v}_i$
- a personal best position $\mathbf{p}_i$
- access to the swarm's best known position $\mathbf{g}$

When a particle finds a position with a better objective value, its personal
best is updated. When any personal best improves on the swarm best, the swarm
best is updated.

## 3) Velocity and position update

The basic PSO update combines inertia, personal learning, and social learning:

$$
v_{i,d} \leftarrow
\omega v_{i,d}
+ \phi_p r_p(p_{i,d} - x_{i,d})
+ \phi_g r_g(g_d - x_{i,d})
$$

$$
\mathbf{x}_i \leftarrow \mathbf{x}_i + \mathbf{v}_i.
$$

The random values $r_p$ and $r_g$ are sampled from $[0,1]$. In `pyswarm`, these
terms map directly to:

- `omega`: inertia weight
- `phip`: cognitive weight, pulling particles toward their own best position
- `phig`: social weight, pulling particles toward the swarm best position

## 4) Bounds

`pyswarm` requires lower and upper bounds:

$$
\mathbf{l_b} \le \mathbf{x} \le \mathbf{u_b}.
$$

Particles are initialized uniformly inside these bounds. During the search, if
a particle moves outside the allowed region, its position is clipped back to the
nearest bound.

## 5) Constraints

Constraints are written as inequality functions. A particle is feasible when
every constraint value is non-negative:

$$
c_j(\mathbf{x}) \ge 0.
$$

There are two supported forms:

- `f_ieqcons`: one function returning a list or array of constraint values
- `ieqcons`: a list of scalar constraint functions

The examples in this project use `f_ieqcons` for both the constrained banana
problem and the two-bar truss problem.

## 6) Parameters and behavior

The choice of PSO parameters affects convergence speed and solution quality.
The Wikipedia summary notes that inertia should be kept below `1` to avoid
unbounded divergence in common analyses, and that cognitive/social coefficients
are often chosen by the practitioner or tuned for the problem.

The defaults in `pyswarm` are conservative:

- `omega = 0.5`
- `phip = 0.5`
- `phig = 0.5`
- `swarmsize = 100`
- `maxiter = 100`

Increasing `swarmsize` explores more candidate solutions per iteration.
Increasing `maxiter` allows a longer search. Adjusting `omega`, `phip`, and
`phig` changes the balance between exploration and exploitation.

## 7) Topology

PSO variants can use different communication topologies. In a global topology,
all particles share the same best known position. In local topologies, each
particle exchanges information with only a subset of neighbors, such as a ring
neighborhood.

`pyswarm` uses the global-best form: all particles are guided by the best
feasible position found by the swarm.

## 8) Stopping criteria

The optimizer stops when one of these conditions is met:

- `maxiter` iterations have been performed
- the best position changes by less than `minstep`
- the best objective value improves by less than `minfunc`

Because PSO is stochastic, repeated runs may return slightly different
solutions.

## Source

This page summarizes the PSO article text supplied from
[Wikipedia: Particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
