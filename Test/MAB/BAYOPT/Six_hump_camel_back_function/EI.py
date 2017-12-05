import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Multi_arm_bandit_algorithms.GP_bandits.Expected_improvement import Expected_improvement

def f(x):
    return np.exp(-(x-2)**2) + np.exp(-(x-6)**2/10) + 1/(x**2+1)

'''
a = np.linspace(-4, 10, 100)
b = [f(i) for i in a]

plt.plot(a,b)
plt.show()
'''
noise = 0.1
x = [-20, -3, 10]
y = [np.random.normal(f(i), noise) for i in x]

u_bound = 10
l_bound = -20


for i in range(30):

    x_, value, y_ = Expected_improvement(x, y, lower_bound=l_bound, upper_bound=u_bound,
                              noise= noise, return_aqs=True, datatype=tf.float32, expected_imporvement_neta=0.05)
    x.append(x_)
    x.sort()
    y = [np.random.normal(f(i), noise) for i in x]
    print(x, y)
    a = np.linspace(l_bound, u_bound, 40)

    v = [i[0][0] for i in value]

    fig, arr = plt.subplots(2, sharex=True)

    arr[0].plot(np.linspace(l_bound, u_bound, 40), v)
    arr[0].set_title("Aqusition Function")
    arr[1].plot(x, y, 'ro')
    arr[0].plot(x_, y_, 'g*')
    fig.savefig(str(i))
    plt.close(fig)
    print(x_)

