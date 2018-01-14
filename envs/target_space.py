import numpy as np
from util.rand import ensure_rng

def _hashable(x):
    """ ensure that an point is hashable """
    return tuple(map(float,x))

class TargetSpace(object):
    def __init__(self, target_func, bounds, random_state=None):
        """
        Paramepers:
        -----------
        target_func: callable
            The objective function must be in the form ``f(x)``
            ``x`` is 2d array or 1d array, 
            must be form [sample_num, feature_dim] or [feature_dim,]
            The return of ``f(x)`` is 2d array or 1d array,
            must be form [sample_num, output_dim] or [output_dim,]

        bounds: tuple or list
            ``(min, max)`` array(shape=(feature_dim,)pairs

        random_state: int, RandomState, None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state=random_state)
        self.target_func = target_func
        
        assert type(bounds[0])==type(bounds[1]), 'the type of lower and upper must be same'
        assert bounds[0].ndim==1, 'ndim of the lower must be 1'

        self.lower = bounds[0]
        self.upper = bounds[1]
        self.bounds = bounds
        self.xdim = self.lower.shape[-1]
       
        self._Xlist = []
        self._Ylist = []

        # Number of observation
        self._length = 0    
        # keep track points of unique points
        self._cache={}
    
    
    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        return self._length

    @property
    def X(self):
        return self._Xlist

    @property
    def Y(self):
        return self._Ylist
    

    def random_inputs(self, points_num):
        """
        Parameters:
        ------------
        points_num: integer
            the number of points selected 
        
        Returns:
        ------------
        xs: 2d-array, [points_num, xdim]

        """
        shape = (points_num,)+self.lower.shape
        xs = self.random_state.uniform(self.lower, self.upper, shape)
        return xs

    def order_inputs(self, num_per_dim):
        """
        Parameters:
        ------------
        num_per_dim: integer
            The number of per dimension

        Returns:
        ------------
        xs: 2d-array, [all_num, x_dim]

        """

        if num_per_dim**self.xdim >10e8:
            num_per_dim = np.floor(np.exp(self.xdim/(8*np.log(10))))

        rang = np.array([np.linspace(0,1,num_per_dim)]).T
        rang = (self.upper-self.lower)*rang+self.lower
        xi = [rang[:,i] for i in range(self.xdim)]
        xs =  np.stack(np.meshgrid(*xi), axis=self.xdim).reshape((-1,self.xdim))
        return xs


    def add_points(self, x, y):
        """
        Parameters:
        ------------
        x: 2d-array or 1d-array
        y: 2d-array or 1d-array
        """

        assert x.ndim==y.ndim, 'x ndim must be same as y ndim'
        assert x.ndim<=2, 'x ndim must be <=2'

        if x.ndim==1 and (x not in self):
            self._cache[_hashable(x)] = y
            self._Xlist.append(x)
            self._Ylist.append(y)
            self._length += 1
        else:
            for xi, yi in zip(x, y):
                if xi not in self:
                    self._cache[_hashable(xi)] = y
                    self._Xlist.append(xi)
                    self._Ylist.append(yi)
                    self._length += 1


    def set_random_state(self,random_state): 
        """
        set random state
        """
        self.random_state = ensure_rng(random_state=random_state)
    

    def cal_output(self,x):
        """
        calculate outputs of the target function 
        Parameters:
        ------------
        x: 2d-array or 1d-array
        """
        assert x.ndim<=2, 'x ndim must <=2'
        return self.target_func(x)
        

    def set_bounds(self, new_bounds):
        """
        Parameters:
        ------------
        new_bounds: dict

        """
        self.bounds = new_bounds
        




