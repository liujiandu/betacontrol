import sys
sys.path.append('../../')
import numpy as np

from envs.target_space import TargetSpace
from envs.function import nlfunc
from regressors.gaussian_process import GP
from acqfuncs.gpacq import AcquisitionFunction

target = TargetSpace(nlfunc, (np.array([0]),np.array([10])), random_state=1)
gp = GP('matern', 25, 1)

acq = AcquisitionFunction('ucb', 1.0, 1.0)
acqkw = {'n_warmup':10, 'n_iter':10}

#ac = acq.acq_max(gp,1, (np.array([0]), np.array([10])),target.random_state, n_warmup=10, n_iter=100)
x=np.array([[1]])
y=np.array([[1]])
gp.fit(x,y)
ac = acq.acq_max(gp,1, (np.array([0]), np.array([10])),target.random_state, **acqkw)
print ac

