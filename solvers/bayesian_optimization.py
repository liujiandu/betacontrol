import numpy as np
from scipy.stats import norm


class BayesianOptimization(object):
    def __init__(self, target_space, regressor, acq, random_state):
        # target function space
        self.space = target_space

        # regressor
        self.regressor = regressor

        # acqusition function
        self.acq = acq

        # random state
        self.random_state = random_state

        self.x_init = []
        self.y_init = []
        self.initialized = False
        self._acqkw = {'n_warmup':1000, 'n_iter':250}

    
    def init(self, init_points):
        #init points
        init_x = self.space.random_inputs(init_points)
        init_y = self.space.cal_output(init_x)
        
        self.space.add_points(init_x, init_y)
        self.initialized = True
    

    def maximize(self, 
                 init_points=5, 
                 n_iter=25, 
                 acq='ucb', 
                 kappa=2.576, 
                 xi=0.0, 
                 **gp_params):
        
        #initialize
        if not self.initialized:
            self.init(init_points)
       
        y_max = max(self.space.Y)

        #set gp parameters
        #self.regressor.set_params(**gp_params) 
    
        #gaussian process fit
        self.regressor.fit(self.X, self.Y)
        
        
        #find argmax of the acquisition function
        x_max = self.acq.acq_max(#ac=self.acq.acqf,
                        regressor = self.regressor,
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
    


