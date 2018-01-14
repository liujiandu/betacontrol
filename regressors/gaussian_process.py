from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class GP(object):
    def __init__(self, kernel_name, n_restarts_optimizer, random_state):
        if kernel_name=='matern':
            self.kernel = Matern(nu=2.5)
        self.gpr_ = GaussianProcessRegressor(kernel=self.kernel,
                n_restarts_optimizer=n_restarts_optimizer,
                random_state = random_state)
    
    def fit(self, x, y):
        """
        gaussian process fit

        Parameters:
        ------------
        x: 2d-array
            must be 2d-array, shape = (smaple_num, feature_dim)
        y: 2d-array
            must be 2d-array, shape = (sample_num, output_dim)

        """
        assert x.ndim==2, 'x must be 2D array'
        assert y.ndim==2, 'y must be 2D array'
        return self.gpr_.fit(x,y)

    def predict(self, x):
        """
        predict by gaussian process fit

        Parameters:
        ------------
        x: 2d-array
            must be 2d-array, shape=[sample_num, feature_dim]
        return_std: bool
        
        Returns:
        mu: 2d-array 
            must be 2d-array, shape = (sample_num, output_dim)
        sigma: 1d-array
            shape = (sample_num,)
        """
        assert x.ndim==2, 'x must be 2D array'
        mu, sigma = self.gpr_.predict(x, return_std=True)
        return mu, sigma

    def set_params(self, **gp_params):
        self.gpr_.set_params(**gp_params)
