from mabandit.Machine_learning.Gaussian_process import Gaussian_process
import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def Probability_of_improvement(X, Y, lower_bound, upper_bound, gaussian_mean = 0,
                         gaussian_variance_weight = 1, gaussian_zigma = 1,
                         noise = None, probability_of_improvement_neta = 0.2,
                         datatype = tf.float64, max_iter = 20, return_aqs = False):

    neta = probability_of_improvement_neta

    G = Gaussian_process(datatype)

    if noise == None:
        G.fit_noiseless(X, Y, noise=noise, variance_weight=gaussian_variance_weight,
                        mean=gaussian_mean, zigma=gaussian_zigma)

    else:
        G.fit_noisy(X, Y, noise= noise, variance_weight=gaussian_variance_weight,
                    mean=gaussian_mean, zigma=gaussian_zigma)

    f_max = max(Y)


    def function(x):
        g_ = G.predict(x)
        r = (g_[0] - f_max - neta)/np.sqrt(g_[1])
        ans = 1- norm.cdf(r)

        return -ans
    '''
    minima = minimize(function, x0= (lower_bound+upper_bound)/2,
                    bounds=[(lower_bound, upper_bound)],
                    options={'maxiter':max_iter, 'disp':False})['x'][0]

    if return_aqs == False:
        return minima
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
        return minima, values, minima_value
    
