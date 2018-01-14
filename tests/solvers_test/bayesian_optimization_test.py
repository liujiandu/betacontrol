import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from solvers.bayesian_optimization import BayesianOptimization
from envs.function import nlfunc
from envs.target_space import TargetSpace                                   
from regressors.gaussian_process import GP as GaussianProcessRegressor
from optimizers.scipy_opt import minimize
from acqfuncs.ucb import UCB
from util.rand import ensure_rng

def plt_gp(bo, x, y):
    fif = plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    bo.regressor.fit(bo.X, bo.Y) 
    mu, sigma = bo.regressor.predict(x)
    mu = mu.flatten() 
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y.flatten(), 'D', markersize=8, color='r', label='Observation')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    axis.plot(x, np.zeros(x.shape[0]),  linewidth=3, color='r', label='Prediction')
    if sigma is not None:
        axis.fill(np.concatenate([x, x[::-1]]), np.concatenate([mu-1.96*sigma, (mu+1.96*sigma)[::-1]]), alpha=0.6, fc='c', ec='None')

    utility = bo.acq.acqf(x)
    acq.plot(x, utility, label='Utility Function')
    plt.show()


if __name__=="__main__":
    random_state = ensure_rng(random_state=1) 
    regressor = GaussianProcessRegressor(kernel_name='matern',
                                         n_restarts_optimizer=25,
                                         random_state=random_state)
    space = TargetSpace(nlfunc, (np.array([-5]), np.array([10])), random_state=random_state)
    acq = UCB(regressor, kind='ucb', kappa=5)

    bo = BayesianOptimization(space, regressor, acq, random_state)
    bo.maximize(init_points=2, n_iter=10, acq='ucb', kappa=5)
    x = np.linspace(-5, 10, 200).reshape(-1,1)
    x = {"x":x}
    y = nlfunc(**x)
    plt_gp(bo, x['x'], y)
