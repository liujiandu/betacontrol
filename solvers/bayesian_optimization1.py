import numpy as np
from scipy.stats import norm

from envs.target_space import TargetSpace
from regressors.mlp import MLP
from optimizers.scipy_opt import minimize
from acqfuncs.identify import Identify
from util.rand import ensure_rng


class BayesianOptimization(object):
    def __init__(self, target_func, bounds, random_state=None):
        self.bounds = bounds

        #random state from random number generate
        self.random_state = ensure_rng(random_state)

        #regressor
        self.space = TargetSpace(target_func, bounds, random_state=random_state)
        self.regressor = MLP(self.space.xdim, 1, [50,30]) 
        
        self.x_init = []
        self.y_init = []
        #self.init_points = []
        self.initialized=False
        self._acqkw = {'n_warmup':1000, 'n_iter':250}

    
    def init(self, init_points):
        #init points
        rand_points = self.space.random_inputs(init_points)
        #self.init_points.append(rand_points)
        
        #evaluate target function at all init points
        #for x in self.init_points:
        y = self.space.cal_output(rand_points)

        #print rand_points
        #print y
        self.space.add_points(rand_points, y)
        self.initialized = True
        

    def maximize(self, 
                 init_points=5, 
                 n_iter=25, 
                 acq='ucb', 
                 kappa=2.576, 
                 xi=0.0, 
                 **gp_params):

        #get acquisition function
        self.acq = Identify(kind=acq)
        
        #initialize
        if not self.initialized:
            self.init(init_points)
       
        y_max = max(self.space.Y)

        #set gp parameters
        #self.regressor.set_params(**gp_params) 
    
        #gaussian process fit
        self.regressor.fit(self.X, self.Y)
        
        
        #find argmax of the acquisition function
        x_max = self.acq.acq_max(regressor = self.regressor,
                                 y_max=y_max,
                                 bounds = self.space.bounds,
                                 random_state = self.random_state,
                                 **self._acqkw)
        #Iterative process
        for i in range(n_iter):
            #Append most recently generated values to X and Y array
            y = self.space.cal_output(x_max)
            self.space.add_points(x_max,y)
            #updatging the regressor
            self.regressor.fit(self.X, self.Y)

            #update maximum value to search for next probe point
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]
            #Maximum acquasition function to find next probing point
            x_max = self.acq.acq_max(regressor = self.regressor,
                                     y_max=y_max,
                                     bounds = self.space.bounds,
                                     random_state = self.random_state,
                                     **self._acqkw)
    @property
    def X(self):
        return np.array(self.space.X)
    @property
    def Y(self):
        return np.array(self.space.Y)
    

if __name__=="__main__":
    from env.function import nlfunc

    bo = BayesianOptimization(nlfunc,(np.array([0,-5]), np.array([10, 10])))
    bo.maximize(init_points=2, n_iter=10, acq='ucb', kappa=5)






