import matplotlib.pyplot as plt
import numpy as np
from Machine_learning.Gaussian_process import Gaussian_process
import time

t = time.time()
G = Gaussian_process()

x = [1,2,3]
y = [3, 10, 2]
G.fit_noiseless(x, y)
print(G.predict(1.5))




print(time.time()-t)