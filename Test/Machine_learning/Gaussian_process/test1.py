import matplotlib.pyplot as plt
import numpy as np
from Machine_learning.Gaussian_process import Gaussian_process
import time

t = time.time()
G = Gaussian_process()

x= [-3, -2, 4]
y= [9, 4, 16]
G.fit_noisy(x, y, noise=0.01)
for i in range(-10, 10):
    print(i)
    print(G.predict(i))




print(time.time()-t)