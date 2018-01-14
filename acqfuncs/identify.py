"""
acqusition funcion is objective function
Author: Liujiandu
Date: 2018/1/19
"""

import numpy as np
from scipy.stats import norm

from optimizers.scipy_opt import minimize


class Identify(object):
    def __init__(self, kind):
        self.kind = kind

    def acqf(self, x, regressor):
        """
        acquisition function
        Parameters:
        ------------
        x: 2d-array, [sample_num, feature_dim]
        regressor: callable
        """

        return self._identify(x, regressor)
    
    @staticmethod
    def _identify(x, regressor):
        """
        mean must be 2d array
        
        Returns:
        -----------
        2d-array, [sample_num, 1]
        """
        mean, _ = regressor.predict(x)
        return mean
    
    def acq_max(self, regressor, y_max, bounds, random_state, n_warmup=100000, n_iter=250):
        #ward up eith random points
        x_tries = random_state.uniform(bounds[0], bounds[1], size=(n_warmup,bounds[0].shape[0]))
        acq_tries = self.acqf(x_tries, regressor)
        x_max = x_tries[np.argmax(acq_tries, axis=0)]
        acq_max = np.max(acq_tries, axis=0)

        #explore the parameter space more throughly
        x_seeds = random_state.uniform(bounds[0], bounds[1], size=(n_iter, bounds[0].shape[0]))
        for x_try in x_seeds:
            res = minimize(lambda x: -self.acqf(x.reshape((-1,1)), 
                                                regressor=regressor),
                           x_try,
                           bounds=zip(bounds[0], bounds[1]),
                           method='L-BFGS-B')

            if acq_max is None or -res.fun[0] >= acq_max:
                x_max = res.x
                acq_max = -res.fun[0]
        return np.clip(x_max, bounds[0], bounds[1])
        
