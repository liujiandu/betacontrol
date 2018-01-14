"""
test function for Optimization algorithm,
or neural network, or other model

Author: Liujiandu
Date: 2018/1/9
"""

import numpy as np


def nlfunc(x):
    """
    target function is a multi-dimensional nonlinear function 

    Parameters:
    -----------
    x: ndarray 
       input of function,[[...],feature_dim]
    
    Retruns:
    ----------
    y: ndarray  
       output of function, [[...], 1]
    """    
    z = np.exp(-(x-2)**2)+np.exp(-(x-6)**2/5)+1/(x**2+1)+0.1*np.sin(5*x)-0.5
    y = np.mean(z, axis=-1)
    return y.reshape(y.shape+(1,))
    

def lfunc(x):
    """
    target function is a multi-dimensional nonlinear function 

    Parameters:
    -----------
    x: ndarray, 
       input of function, [[...],feature_dim]
    
    Retruns:
    ----------
    y: ndarray,
       output of function, [[...],1]
    """    
    z = x
    y = np.mean(z, axis=1)
    return y.reshape(y.shape+(1,))



if __name__=="__main__":
    x = np.array([1,2,3,4])
    assert x.shape==(4,)
    x={"x":x}
    y = nlfunc(**x)
    assert y.shape==(1,)

    x = np.array([[1,2,3,4],
                  [2,3,4,5]])
    assert x.shape==(2,4)
    x={"x":x}
    y = nlfunc(**x)
    assert y.shape==(2,1)
