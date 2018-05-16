import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mabandit.Multi_arm_bandit_algorithms.GP_bandits.Probability_of_improvement import Probability_of_improvement

def f(x):
    return -(x**2 + 10*np.sin(x))

noise = 0.1
x = [-20, -3, 10]
#x = [-20, -14, 10]
y = [np.random.normal(f(i), noise) for i in x]

u_bound = 10
l_bound = -20


for i in range(25):

    x_, values, y_ = Probability_of_improvement(x, y, lower_bound=l_bound, upper_bound=u_bound,
                                    noise=noise, probability_of_improvement_neta=0.3, max_iter=20, return_aqs=True)

    x.append(x_)
    x.sort()
    y = [np.random.normal(f(i), noise) for i in x]

    a = np.linspace(l_bound, u_bound, 40)
    v = [i[0][0] for i in values]

    fig, arr = plt.subplots(2, sharex=True)

    arr[0].plot(np.linspace(l_bound, u_bound, 40), v)
    arr[0].set_title("Aqusition Function")
    arr[1].plot(x, y, 'ro')
    arr[0].plot(x_, y_, 'g*')
    fig.savefig(str(i))
    plt.close(fig)

    print(x_)
