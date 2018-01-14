import numpy as np
from scipy.stats import norm

from optimizers.scipy_opt import minimize


class AcquisitionFunction(object):
    def __init__(self, name, regressor):
        self.name = name
        self.regressor = regressor

    def acqf(self, x, y_max):
        """
        acquisition function
        Parameters:
        ------------
        x: 2d-array, [sample_num, feature_dim]
        regressor: callable
        """
        pass 


    def acq_max(self, y_max, bounds, random_state, n_warmup=100000, n_iter=250):
        #ward up eith random points
        x_tries = random_state.uniform(bounds[0], bounds[1], size=(n_warmup,bounds[0].shape[0]))
        acq_tries = self.acqf(x_trie, y_max=y_max)
        x_max = x_tries[np.argmax(acq_tries, axis=0)]
        acq_max = np.max(acq_tries, axis=0)

        #explore the parameter space more throughly
        x_seeds = random_state.uniform(bounds[0], bounds[1], size=(n_iter, bounds[0].shape[0]))
        for x_try in x_seeds:
            res = minimize(lambda x: -self.acqf(x.reshape((-1,1)),
                                                y_max=y_max),
                           x_try,
                           bounds=zip(bounds[0], bounds[1]),
                           method='L-BFGS-B')

            if acq_max is None or -res.fun[0] >= acq_max:
                x_max = res.x
                acq_max = -res.fun[0]
        return np.clip(x_max, bounds[0], bounds[1])
        
