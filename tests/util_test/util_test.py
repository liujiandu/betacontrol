"""
test util.rand
Author: Liujiandu
Date: 2018/1/14
"""
import sys
sys.path.append('../../')
import numpy as np

from util.rand import ensure_rng

random_state = ensure_rng(1)

lower = np.array([1,2,3])
upper = np.array([1.5,15,150])

print random_state.uniform(lower, upper, (2,2,3))
