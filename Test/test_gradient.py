from Machine_learning.Beta_bernoulli_model import Beta_bernoulli
import numpy as np
import matplotlib.pyplot as plt

b = Beta_bernoulli()

x = np.linspace(0, 1, 100)
y = [b.PDF(i) for i in x]

plt.plot(x, y)
plt.show()

b.update(0)
b.update(1)

x = np.linspace(0, 1, 100)
y = [b.PDF(i) for i in x]

plt.plot(x, y)
plt.show()




