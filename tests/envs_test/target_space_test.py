#-*-coding=utf-8-*-

"""
envs/target.py test
Author: Liujiandu
Date: 2018/1/13
"""

import sys
sys.path.append('../../')
import numpy as np

from envs.function import nlfunc
from envs.target_space import TargetSpace

def f(x):
    """
    ndim of ``x`` must be same ``f(x)``
    """
    return x


def test(target_space):
    print ('xdim', target_space.xdim)
    print ('length: ', len(target_space))
    print ('random_state: ', target_space.random_state)
    print ('lower: ', target_space.lower)
    print ('upper: ', target_space.upper)
    print ('bounds: ', target_space.bounds)

    x = np.arange(target_space.xdim)
    print ('x: ', x)
    y= target_space.cal_output(x)
    print ("y :", y)

    target_space.add_points(x, y)
    print ("random inputs: ", target_space.random_inputs(2))
    print ("order inputs: ", target_space.order_inputs(3))
    print ("X: ", target_space.X)
    print ("Y: ", target_space.Y)

bounds = (np.array([0,2]),np.array([10,100]))
target_space = TargetSpace(f, bounds, random_state=1)

test(target_space)
