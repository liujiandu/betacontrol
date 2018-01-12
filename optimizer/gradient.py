from scipy import optimize

def minimize(x, x_try, bounds, method):
    return optimize.minimize(x, x_try, bounds=bounds, method=method)
