import numpy as np
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

from env.target_space import TargetSpace
from regressor.gaussian_process import GPR as GaussianProcessRegressor
from optimizer.gradient import minimize
from acqfunc.gpacq import AcquisitionFunction
from util.rand import ensure_rng


class BayesianOptimization(object):
    def __init__(self, f, pbounds, random_state=None):
        self.pbounds = pbounds

        #random state from random number generate
        self.random_state = ensure_rng(random_state)

        #regressor
        self.regressor = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                           n_restarts_optimizer=25,
                                           random_state=self.random_state)
        self.init_points = []
        self.space = TargetSpace(f, pbounds, random_state)
        
        self.x_init = []
        self.y_init = []
        self.initialized=False
        self._acqkw = {'n_warmup':100000, 'n_iter':250}

    def init(self, init_points):
        #init points
        rand_points = self.space.random_points(init_points)
        self.init_points.extend(rand_points)

        #evaluate target function at all init points
        for x in self.init_points:
            y = self._observe_point(x)

        #add the points to the obsevations
        if self.x_init:
            x_init = np.vstack(self.x_init)
            y_init = np.hstack(self.y_init)
            for x, y in zip(x_init, y_init):
                self.space.add_observation(x,y)
        self.initialized = True

    def _observe_point(self, x):
        y = self.space.observe_point(x)
        return y


    def initialize(self, points_dict):
        self.y_init.extend(points_dict['target'])
        for i in range(len(points_dict['target'])):
            all_points = []
            for key in self.space.keys:
                all_points.append(points_dict[key][i])
            self.x_init_append(all_points)


    def maximize(self, 
                 init_points=5, 
                 n_iter=25, 
                 acq='ucb', 
                 kappa=2.576, 
                 xi=0.0, 
                 **gp_params):

        #get acquisition function
        self.acq = AcquisitionFunction(kind=acq, kappa=kappa, xi=xi)
        
        #initialize
        if not self.initialized:
            self.init(init_points)
        y_max = self.space.Y.max()

        #set gp parameters
        self.regressor.set_params(**gp_params)

        #gaussian process fit
        self.regressor.fit(self.space.X, self.space.Y)

        #find argmax of the acquisition function
        x_max = self.acq.acq_max(ac=self.acq.acqf,
                        gp = self.regressor,
                        y_max=y_max,
                        bounds = self.space.bounds,
                        random_state = self.random_state,
                        **self._acqkw)
        
        #Iterative process
        for i in range(n_iter):
            #Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            #updatging the regressor
            self.regressor.fit(self.space.X, self.space.Y)

            #update maximum value to search for next probe point
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]
            #Maximum acquasition function to find next probing point
            x_max = self.acq.acq_max(ac=self.acq.acqf,
                            gp = self.regressor,
                            y_max=y_max,
                            bounds = self.space.bounds,
                            random_state = self.random_state,
                            **self._acqkw)
    @property
    def X(self):
        return self.space.X
    @property
    def Y(self):
        return self.space.Y

if __name__=="__main__":
    from env.function import target

    bo = BayesianOptimization(target, {'x':(-5, 10)})
    bo.maximize(init_points=2, n_iter=10, acq='ucb', kappa=5)






