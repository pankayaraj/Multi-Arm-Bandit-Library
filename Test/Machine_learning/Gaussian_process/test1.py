import matplotlib.pyplot as plt
import numpy as np
from Machine_learning.Gaussian_process import Gaussian_process
import time

t = time.time()
G = Gaussian_process()

y = [np.random.randint(2, 20) for i in range(15)]
x = np.linspace(0, 20, num=15)
G.fit_noisy(x, y, 0.1)
print(G.predict(21))



print(time.time()-t)