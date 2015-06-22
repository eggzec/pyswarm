import logging
import numpy as np

__all__ = ['logger', 'pso']

logger = logging.getLogger('pso')
s = logging.StreamHandler()
f = logging.Formatter('%(name)s [%(levelname)s]: %(message)s')
s.setFormatter(f)
logger.addHandler(s)


def no_constraint(position):
    return True


def pso(objectivefunc, lowbound, upbound, constraintfunc=no_constraint,
        swarmsize=10, omega=0.5, phip=0.5, phig=0.5, maxiters=1000,
        minstep=1e-8, minfunc=1e-8, funcargs=None, funckwargs=None):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    objectivefunc : function
        The function to be minimized
    lowbound : array
        The lower bounds of the design variable(s)
    upbound : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    constraintfunc : function
        Returns True if the particle position in question fits the constraints.
        Returns False otherwise. (Default: no_constraint)
    funcargs : tuple
        Additional arguments passed to objective and constraint functions
        (Default: None)
    funckwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: None)
    swarmsize : int
        The number of particles in the swarm (Default: 10)
    omega : scalar
        Particle's inertia weight (Default: 0.5)
    phip : scalar
        Particle's cognitive weight (Default: 0.5)
    phig : scalar
        Swarm's social weight (Default: 0.5)
    maxiters : int
        The maximum number of iterations for the swarm to search (Default: 1000)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
   
    """

    lowbound = np.array(lowbound)
    upbound = np.array(upbound)
    assert len(lowbound) == len(upbound), 'Lower- and upper-bounds must be the same length'
    assert hasattr(objectivefunc, '__call__'), 'Invalid function handle'
    assert np.all(upbound > lowbound), 'All upper-bound values must be greater than lower-bound values'

    if funcargs is None:
        funcargs = ()
    if funckwargs is None:
        funckwargs = {}

    vhigh = np.abs(upbound - lowbound)
    vlow = -vhigh

    def obj(position):
        return objectivefunc(position, *funcargs, **funckwargs)

    def is_feasible(position):
        return constraintfunc(position, *funcargs, **funckwargs)

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lowbound)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value
    
    for i in xrange(S):
        # Initialize the particle's position
        x[i, :] = lowbound + x[i, :]*(upbound - lowbound)
   
        # Initialize the particle's best known position
        p[i, :] = x[i, :]
       
        # Calculate the objective's value at the current particle's
        fp[i] = obj(p[i, :])
       
        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i] < fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()
       
        # Initialize the particle's velocity
        v[i, :] = vlow + np.random.rand(D)*(vhigh - vlow)
       
    # Iterate until termination criterion met ##################################
    for it in range(maxiters):
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        for i in xrange(S):

            # Update the particle's velocity
            v[i, :] = omega*v[i, :] + phip*rp[i, :]*(p[i, :] - x[i, :]) + \
                      phig*rg[i, :]*(g - x[i, :])
                      
            # Update the particle's position, correcting lower and upper bound 
            # violations, then update the objective function value
            x[i, :] = x[i, :] + v[i, :]
            mark1 = x[i, :] < lowbound
            mark2 = x[i, :] > upbound
            x[i, mark1] = lowbound[mark1]
            x[i, mark2] = upbound[mark2]
            fx = obj(x[i, :])
            
            # Compare particle's best position (if constraints are satisfied)
            if fx < fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx < fg:
                    logger.debug('New best for swarm at iteration {}: {} '
                                 '{}'.format(it, x[i, :], fx))
                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g - tmp)**2))
                    if np.abs(fg - fx) <= minfunc:
                        logger.debug('Stopping search: Swarm best objective '
                                     'change less than: {}'.format(minfunc))
                        return tmp, fx
                    elif stepsize <= minstep:
                        logger.debug('Stopping search: Swarm best position '
                                     'change less than: {}'.format(minstep))
                        return tmp, fx
                    else:
                        g = tmp.copy()
                        fg = fx
        logger.debug('Best after iteration {} {} {}'.format(it, g, fg))
    logger.debug('Stopping search: maximum iterations reached --> '
                 '{}'.format(maxiters))
    if g is []:
        logger.warning('No feasible point found')
    return g, fg

