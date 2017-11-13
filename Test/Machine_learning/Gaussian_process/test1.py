import matplotlib.pyplot as plt
import numpy as np
from Machine_learning.Gaussian_process import Gaussian_process

G = Gaussian_process()

y = [np.random.randint(2, 20) for i in range(10)]
x = np.linspace(0, 20, num=10)
G.fit_noiseless(x, y)
print(G.predict(21))
fig1 = plt.plot(x, y)
plt.show()