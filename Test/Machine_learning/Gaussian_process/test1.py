import matplotlib.pyplot as plt
import numpy as np
from Machine_learning.Gaussian_process import Gaussian_process
import time

t = time.time()
G = Gaussian_process()

x= [-3, -2, 4]
y= [9, 4, 10]


G.fit_noisy(x, y, noise=0.01, variance_weight=1)
fig = plt.figure("A")
for _ in range(50):
    x = np.linspace(-5,5, 20)
    y = [0]*len(x)
    for i in range(len(x)):
        g_ = G.predict(x[i])
        y[i] = (g_[0] + np.random.normal()*g_[1])[0]

    plt.plot(x, y, 'b')



plt.show()

print(time.time()-t)