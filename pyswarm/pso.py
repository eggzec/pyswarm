import logging
import warnings
import numpy as np

logger = logging.getLogger('pso')


def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=10, omega=0.5, phip=0.5, phig=0.5, maxiter=1000, 
        minstep=1e-8, minfunc=1e-8):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
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
        The number of particles in the swarm (Default: 10)
    omega : scalar
        Particle's inertia weight (Default: 0.5)
    phip : scalar
        Particle's cognitive weight (Default: 0.5)
    phig : scalar
        Swarm's social weight (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 1000)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
   
    """

    lb = np.array(lb)
    ub = np.array(ub)
    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Check for constraint function(s) #########################################
    obj = lambda (x): func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            logger.debug('No constraints given.')
            cons = lambda (x): np.array([0])
        else:
            logger.debug('Converting ieqcons to a single constraint function')
            cons = lambda (x): np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        logger.debug('Single constraint function given in f_ieqcons')
        cons = lambda (x): np.array(f_ieqcons(x, *args, **kwargs))
        
    def is_feasible(x):
        check = np.all(cons(x) >= 0)
        return check
        
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value
    
    for i in xrange(S):
        # Initialize the particle's position
        x[i, :] = lb + x[i, :]*(ub - lb)
   
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
    it = 1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        for i in xrange(S):

            # Update the particle's velocity
            v[i, :] = omega*v[i, :] + phip*rp[i, :]*(p[i, :] - x[i, :]) + \
                      phig*rg[i, :]*(g - x[i, :])
                      
            # Update the particle's position, correcting lower and upper bound 
            # violations, then update the objective function value
            x[i, :] = x[i, :] + v[i, :]
            mark1 = x[i, :] < lb
            mark2 = x[i, :] > ub
            x[i, mark1] = lb[mark1]
            x[i, mark2] = ub[mark2]
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
        it += 1
    logger.debug('Stopping search: maximum iterations reached --> '
                 '{}'.format(maxiter))
    if g is []:
        warnings.warn('No feasible point found')
    return g, fg

