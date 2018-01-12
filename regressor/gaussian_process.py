from sklearn.gaussian_process import GaussianProcessRegressor


class GPR(object):
    def __init__(self, kernel, n_restarts_optimizer, random_state):
        self.gpr_ = GaussianProcessRegressor(kernel=kernel,
                n_restarts_optimizer=n_restarts_optimizer,
                random_state = random_state)
    
    def fit(self, x, y):
        return self.gpr_.fit(x,y)

    def predict(self, x, return_std=False):
        return self.gpr_.predict(x, return_std=return_std)

    def set_params(self, **gp_params):
        self.gpr_.set_params(**gp_params)
