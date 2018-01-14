"""
test envs/function.py
Author: Liujiandu
Date: 2018/1/14
"""
import sys
sys.path.append('../../')

import numpy as np
from envs.function import nlfunc

x = np.array([[1,2,3]])
y = nlfunc(x)
print y

