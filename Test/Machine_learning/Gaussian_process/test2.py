import sklearn.gaussian_process as gp
import time
import numpy as np
import matplotlib.pyplot as plt
t = time.time()


kernel = gp.kernels.RBF()
model = gp.GaussianProcessRegressor(kernel=kernel,
                                      )

y = np.reshape([np.random.randint(2, 20) for i in range(15)], (15,1))
x = np.reshape(np.linspace(0, 20, num=15), (15,1))

model.fit(x, y)
print(model.predict(21, return_std=True))

print(time.time()-t)


X = np.linspace(-100, 100, 200)
Y = [x**3 - 2*x + 3*x**2 for x in X]

plt.plot(X,Y)
plt.show()