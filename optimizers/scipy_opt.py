from scipy import optimize

def minimize(func, x0, bounds=None, method=None, maxiter=100, disp=False):
    """
    wrap scipy.optimize.minimize
    
    Parameters:
    ------------
    func: callable
        The objective function to be minimized. 
        Must be in the form ``f(x)``, x is a ndarray.
    x0: ndarray
        Initial guess, the dimension of ``x0`` is the same as 
        the input of ``func``
    bounds: sequence, optional
        Bounds for variables(only for L-BFGS-B, TNC, SLSQP)
        ``(min, max)`` pairs for each element in ``x``

    method: str
        Should be one of
            'Nelder-Mead',  #0                                                                 
            'Powell',       #1
            'CG',           #2
            'BFGS',         #3
            'Newton-CG',    #4
            'TNC',          #5
            'COBYLA',       #6
            'SLSQP',        #7
            'dogleg',       #8
            'trust-ncg',    #9
            'trust-exact',  #10
    options:
    """

    METHOD=['Nelder-Mead',  #0                                                                 
            'Powell',       #1
            'CG',           #2
            'BFGS',         #3
            'Newton-CG',    #4
            'L-BFGS-B',     #5
            'TNC',          #6
            'COBYLA',       #7
            'SLSQP',        #8
            'dogleg',       #9
            'trust-ncg',    #10
            'trust-exact',  #11
            'trust-krylov'  #12
           ]
    assert method in METHOD, '[{}] method not in scipy optimize!'.format(method)
    if method not in ['L-BFGS-B', 'TNC', 'SLSQP']:
        assert bounds is None, 'Bounds only for L-BFGS-B, TNC, SLSQP'

    return optimize.minimize(func, x0, bounds=bounds, method=method, options={'maxiter':maxiter, 'disp':disp})


