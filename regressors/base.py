"""
regressor base class
Author: Liujiandu
Date: 2018/1/14
"""

class BaseRegressor(object):
    def __init__(self, name):
        self.name = name

    def fit(self, x, y):
        return NotImplemented
    
    def predict(self, x):
        return NotImplemented
