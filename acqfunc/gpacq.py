import numpy as np
from scipy.stats import norm

from optimizer.gradient import minimize


class AcquisitionFunction(object):
    def __init__(self, kind, kappa, xi):
        self.kappa = kappa
        self.xi = xi
        if kind not in ['ucb', 'ei', 'poi']:
            err = 'the utility function {} has not been implement'.format(kind)
            raise NotImplementedError()
        self.kind = kind

    def acqf(self, x, gp, y_max):
        if self.kind=="ucb":
            return self._ucb(x, gp, self.kappa)
        if self.kind=="ei":
            return self._ei(x, gp, y_max, self.xi)
        if self.kind=="poi":
            return self._poi(x, gp, y_max, self.xi)
    
    @staticmethod
    def _ucb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean +kappa*std
    
    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi)*norm.cdf(z) +std*norm.pdf(z)
    
    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean -y_max -xi)/std
        return norm.cdf(z)

    def acq_max(self, ac, gp, y_max, bounds, random_state, n_warmup=100000, n_iter=250):
        #ward up eith random points
        x_tries = random_state.uniform(bounds[:,0], bounds[:,1], size=(n_warmup,bounds.shape[0]))
        ys = ac(x_tries, gp, y_max=y_max)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
    
        #explore the parameter space more throughly
        x_seeds = random_state.uniform(bounds[:,0], bounds[:,1], size=(n_iter, bounds.shape[0]))
        for x_try in x_seeds:
            res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method='L-BFGS-B')
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
        return np.clip(x_max, bounds[:,0], bounds[:,1])

