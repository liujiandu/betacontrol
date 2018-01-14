"""
test regressors/gaussian_process.py
Author: Liujiandu
Date: 2018/1/15
"""

import sys
sys.path.append('../../')
import numpy as np

from regressors.gaussian_process import GP

gp = GP('matern', 10, random_state=1)
x = np.ones((10,10))
y = np.ones((10,1))
gp.fit(x,y)
mu, std= gp.predict(x)
print mu.shape
print mu.shape
