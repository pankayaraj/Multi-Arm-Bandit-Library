from scipy.stats import norm
from scipy.optimize import minimize
import tensorflow as tf
import numpy as np
from Machine_learning.Gaussian_process import Gaussian_process

def Expected_improvement(X, Y, lower_bound, upper_bound, gaussian_mean = 0,
                         gaussian_variance_weight = 1, gaussian_zigma = 1,
                         noise = None, expected_imporvement_neta = 0.01,
                         datatype = tf.float64, max_iter = 20, return_aqs = False):

    G = Gaussian_process(datatype=datatype)
    if noise == None:
        G.fit_noiseless(X, Y, noise=noise, variance_weight=gaussian_variance_weight,
                        mean=gaussian_mean, zigma=gaussian_zigma)

    else:
        G.fit_noisy(X, Y, noise= noise, variance_weight=gaussian_variance_weight,
                    mean=gaussian_mean, zigma=gaussian_zigma)


    f_max= max(Y)
    neta = expected_imporvement_neta

    def function(x):
        g_ = G.predict(x)
        g_s = np.sqrt(g_[1])
        if g_s > 0:
            Z = (g_[0] - f_max - neta)/g_s
        else:
            Z = 0

        if g_s == 0:
            return  0
        else:
            ans = ((g_[0] - f_max - neta)*norm.cdf(Z) + g_s*norm.pdf(Z))
            return ans
    '''
    add a - to the return value when using minimise function for maximization
    minima = minimize(function,method= 'TNC', x0= (lower_bound+upper_bound)/2,
                    bounds=[(lower_bound, upper_bound)],
                    options={'maxiter':max_iter, 'disp':False})['x'][0]
    '''
    a = np.linspace(lower_bound, upper_bound, 40)
    values = [function(i) for i in a]

    minima = a[0]
    minima_value = values[0]
    for i in range(40):
        if values[i] > minima_value:
            minima = a[i]
            minima_value = values[i]



    if return_aqs == False:

        return minima

    else:
        x = np.linspace(lower_bound, upper_bound, 40)
        y = [function(i) for i in x]
        minima_value = function(minima)

        return minima, y, minima_value
