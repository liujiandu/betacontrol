import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

#from bayes_opt import BayesianOptimization
from bayesian_optimization import BayesianOptimization
from env.function import nlfunc


def plt_gp(bo, x, y):
    fif = plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    bo.regressor.fit(bo.X, bo.Y) 
    mu, sigma = bo.regressor.predict(x, return_std=True)
    
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, color='r', label='Observation')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    axis.plot(x, np.zeros(x.shape[0]),  linewidth=3, color='r', label='Prediction')
    axis.fill(np.concatenate([x, x[::-1]]), np.concatenate([mu-1.96*sigma, (mu+1.96*sigma)[::-1]]), alpha=0.6, fc='c', ec='None')

    utility = bo.acq.acqf(x, bo.regressor, 0)
    acq.plot(x, utility, label='Utility Function')
    plt.show()


if __name__=="__main__":
    bo = BayesianOptimization(nlfunc, {'x':(-5,10)})
    bo.maximize(init_points=2, n_iter=10, acq='ucb', kappa=5)
    x = np.linspace(-5, 10, 200).reshape(-1,1)
    x = {"x":x}
    y = nlfunc(**x)
    plt_gp(bo, x['x'], y)
