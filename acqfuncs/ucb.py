import numpy as np
from scipy.stats import norm

from optimizers.scipy_opt import minimize


class UCB(object):
    def __init__(self, regressor, kind, kappa):
        self.regressor = regressor
        self.kappa = kappa
        self.kind = kind

    def acqf(self, x):
        """
        acquisition function
        Parameters:
        ------------
        x: 2d-array, [sample_num, feature_dim]
        regressor: callable
        """

        mean, std = self.regressor.predict(x)
        assert std is not None, 'standard variance is None'
        std = std.reshape((-1,1))
        return mean +self.kappa*std

        #return self._ucb(x, regressor, self.kappa)
    '''
    @staticmethod
    def _ucb(x, regressor, kappa):
        """
        mean must be 2d array
        
        Returns:
        -----------
        2d-array, [sample_num, 1]
        """
        mean, std = regressor.predict(x, return_std=True)
        assert std is not None, 'standard variance is None'
        std = std.reshape((-1,1))
        return mean +kappa*std
    '''

    def acq_max(self, bounds, random_state, n_warmup=100000, n_iter=250):
        #ward up eith random points
        x_tries = random_state.uniform(bounds[0], bounds[1], size=(n_warmup,bounds[0].shape[0]))
        acq_tries = self.acqf(x_tries)
        x_max = x_tries[np.argmax(acq_tries, axis=0)]
        acq_max = np.max(acq_tries, axis=0)

        #explore the parameter space more throughly
        x_seeds = random_state.uniform(bounds[0], bounds[1], size=(n_iter, bounds[0].shape[0]))
        for x_try in x_seeds:
            res = minimize(lambda x: -self.acqf(x.reshape((-1,1))),
                           x_try,
                           bounds=zip(bounds[0], bounds[1]),
                           method='L-BFGS-B')

            if acq_max is None or -res.fun[0] >= acq_max:
                x_max = res.x
                acq_max = -res.fun[0]
        return np.clip(x_max, bounds[0], bounds[1])
        
