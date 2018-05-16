from mabandit.Machine_learning.Gaussian_process import Gaussian_process
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

def Thomspason_sampling(X, Y, lower_bound, upper_bound, gaussian_mean = 0, kernel_zigma = 1, gaussian_variance_weight= 1, datatype= tf.float64,
                        noise = None, max_iter = 20, return_aqs = False):
    G = Gaussian_process(datatype=datatype)

    if noise == None:
        G.fit_noiseless(X, Y, mean = gaussian_mean, zigma = kernel_zigma, variance_weight= gaussian_variance_weight)
    else:
        G.fit_noisy(X, Y, mean= gaussian_mean, zigma= kernel_zigma, variance_weight= gaussian_variance_weight, noise = noise)


    if return_aqs == False:
        def function(x):
            g_ = G.predict(x)
            return -(g_[0] + np.random.normal(0, 1)*g_[1])

        max_x = minimize(function,
                         x0=-80,
                         bounds=[(lower_bound, upper_bound)],
                         options={'maxiter':max_iter, 'disp':False})['x'][0]


        return max_x

    else:
        def function(x):
            g_ = G.predict(x)
            a = -(g_[0] + np.random.normal(0,1)*g_[1])
            fh = open('aqs_function_thompon_sampling', 'a')
            fh.write(str(a[0][0]) + ',' + str(x[0]) + '/n')
            fh.close()

            return a
        max_x = minimize(function,
                         x0=(lower_bound + upper_bound) / 2,
                         bounds=[(lower_bound, upper_bound)],
                         options={'maxiter': max_iter, 'disp': False})['x'][0]

        return max_x
