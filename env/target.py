import numpy as np
from util.rand import ensure_rng

def _hashable(x):
    """ ensure that an point is hashable """
    return tuple(map(float,x))

class Target(object):
    """
    """
    def __init__(self, target_func, pbounds, random_state=None):
        """
        Paramepers:
        -----------
        target_func: 
        pbounds: dict
            Dictionary with parameters names as keys and 
            a tuple with minimum and maximum array or list values
        random_state: int, RandomState, None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state=random_state)
        self.target_func = target_func
        self.pbounds = pbounds  
        self.keys = pbounds.keys()
        self.input_dim = sum([lower.shape[0] for key,(lower, upper) in pbounds.items()])
       
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
        X: dict

        """
        X = {}
        for key in self.keys:
            (lower, upper) = self.pbounds[key]
            assert type(lower)==type(upper) and type(lower)==np.ndarray
            assert lower.shape==upper.shape
            shape = (points_num,)+lower.shape
            rand = self.random_state.uniform(0,1,shape)
            X[key] = rand*(upper-lower)+lower
        
        return X

    def order_inputs(self, num_per_dim):
        """
        Parameters:
        ------------
        num_per_dim:

        Returns:
        ------------
        X: dict
            Dictionary with parameters names as keys and array values

        """

        if num_per_dim**self.input_dim >10e8:
            num_per_dim = np.floor(np.exp(self.input_dim/(8*np.log(10))))

        X = {}
        for key in self.keys:
            (lower, upper) = self.pbounds[key]
            assert type(lower)==type(upper) and type(lower)==np.ndarray
            assert lower.shape==upper.shape
            dim = lower.shape[0]
            rang = np.ones(dim)*np.array([np.linspace(0,1,num_per_dim)]).T
            rang = rang*(upper-lower)+lower
            xi = [rang[:,i] for i in range(dim)]
            X[key] =  np.stack(np.meshgrid(*xi), axis=dim).reshape((-1,dim))
        return X


    def add_point(self,y, **param):
        """
        Parameters:
        ------------
        param: dict
            Dictionary with parameters names as keys and array values            
        y: array
        """
        x = self.dict_to_array(**param)
        if x in self:
            self._cache[_hashable(x)] = y
            self._Xlist.append(x)
            self._Ylist.append(y)
            self._length += 1


    def set_random_state(self,random_state): 
        """
        set random state
        """
        self.random_state = ensure_rng(random_state=random_state)
    

    def cal_output(self, **params):
        """
        calculate outputs of the target function 
        """
        return self.target_func(**params)


    def set_bounds(self, new_pbounds):
        """
        Parameters:
        ------------
        new_bounds: dict

        """
        for row, key in enumerate(self.keys):
            if key in new_pbounds:
                self.pbounds[key] = new_pbounds[key]
        

    def dict_to_array(self, **params):
        """
        dict tranpose to array
        """
        return np.concatenate([params[key] for key in self.keys])




