import sys
sys.path.append('../')
import numpy as np

from env.target import Target
from env.function import nlfunc
from env.target_space import TargetSpace

def f(x, x1):
    return x+x1

pbounds = {'x':(np.array([0,2]),np.array([10,100])),
          'x1':(np.array([0,2]),np.array([10,100]))}
target = Target(f, pbounds, random_state=1)

print ('length: ', len(target))
print ('random_state: ', target.random_state)
print ('keys: ', target.keys)
print ('pbounds: ', target.pbounds)
print ('input_dim', target.input_dim)
x = np.array([1,2])
print ('x',x)
params = {'x':x, 'x1':x}
print ("params: ", params)
print ("dict to array: ", target.dict_to_array(**params))

y = target.cal_output(**params)
print ('y',y)

print ("random inputs: ", target.random_inputs(2))
print ("order inputs: ", target.order_inputs(3))


