import sys
sys.path.append('../../')
import numpy as np

from envs.function import nlfunc
from optimizers.scipy_opt import minimize

def fun(x):
    return sum(x**2).reshape((-1,1))

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
print minimize(fun, np.array([5,5]),bounds=((-10,10),(-10,-3),), method=METHOD[5], maxiter=10)


