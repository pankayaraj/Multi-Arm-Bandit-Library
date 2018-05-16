import matplotlib.pyplot as plt
import numpy as np
from mabandit.Multi_arm_bandit_algorithms.Baysian_bandits.Thomspson_sampling import Thomspason_sampling
import tensorflow as tf

def f(x):
    return -(x**2 + 10*np.sin(x))

noise = 0.1
x = [-10, -4, 5]
y = [np.random.normal(f(i), noise) for i in x]

u_bound = 10
l_bound = -10

for i in range(50):
    x_ = Thomspason_sampling(x, y, lower_bound=-l_bound, upper_bound=u_bound,
                            noise= noise, return_aqs=False, datatype=tf.float32)

    x.append(x_)
    x.sort()
    y = [np.random.normal(f(i), noise) for i in x]
    print(x_)

