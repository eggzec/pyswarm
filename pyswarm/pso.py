from functools import partial
import numpy as np

def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)

def _is_feasible_wrapper(func, x):
    return np.all(func(x)>=0)

def _cons_none_wrapper(x):
    return np.array([0])

def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])

def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))
    
def pso(func, bounds, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
          swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
          minstep=1e-12, minfunc=1e-12, debug=False):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    bounds: tuple array
        The bounds of the design variable(s). In form [(lower, upper), ..., (lower, upper)]
   
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
   
    """
    lower_bound, upper_bound = [], []
    for variable_bounds in bounds:
        lower_bound.append(variable_bounds[0])
        upper_bound.append(variable_bounds[1])

    assert len(lower_bound) == len(upper_bound), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    assert np.all(upper_bound > lower_bound), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.abs(upper_bound - lower_bound)
    vlow = -vhigh

    # Check for constraint function(s) #########################################
    obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = lambda x: np.array([0])
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))

    def is_feasible(x):
        check = np.all(cons(x) >= 0)
        return check

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lower_bound)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value

    for i in range(S):
        # Initialize the particle's position
        x[i, :] = lower_bound + x[i, :] * (upper_bound - lower_bound)

        # Initialize the particle's best known position
        p[i, :] = x[i, :]

        # Calculate the objective's value at the current particle's
        fp[i] = obj(p[i, :])

        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        if i == 0:
            g = p[0, :].copy()

        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i] < fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()

        # Initialize the particle's velocity
        v[i, :] = vlow + np.random.rand(D) * (vhigh - vlow)

    # Iterate until termination criterion met ##################################
    iteration_termination = 1
    while iteration_termination <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        for i in range(S):

            # Update the particle's velocity
            v[i, :] = omega * v[i, :] + phip * rp[i, :] * (p[i, :] - x[i, :]) + \
                      phig * rg[i, :] * (g - x[i, :])

            # Update the particle's position, correcting lower and upper bound
            # violations, then update the objective function value
            x[i, :] = x[i, :] + v[i, :]
            mark1 = x[i, :] < lower_bound
            mark2 = x[i, :] > upper_bound
            x[i, mark1] = lower_bound[mark1]
            x[i, mark2] = upper_bound[mark2]
            fx = obj(x[i, :])

            # Compare particle's best position (if constraints are satisfied)
            if fx < fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx < fg:
                    if debug:
                        print('New best for swarm at iteration {:}: {:} {:}'.format(iteration_termination, x[i, :], fx))

                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g - tmp) ** 2))
                    if np.abs(fg - fx) <= minfunc:
                        print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                        return tmp, fx
                    elif stepsize <= minstep:
                        print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                        return tmp, fx
                    else:
                        g = tmp.copy()
                        fg = fx

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(iteration_termination, g, fg))
        iteration_termination += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    return g, fg
