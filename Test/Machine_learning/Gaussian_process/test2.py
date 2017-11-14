import sklearn.gaussian_process as gp
import time
import numpy as np

t = time.time()


kernel = gp.kernels.RBF()
model = gp.GaussianProcessRegressor(kernel=kernel,
                                      )

y = np.reshape([np.random.randint(2, 20) for i in range(15)], (15,1))
x = np.reshape(np.linspace(0, 20, num=15), (15,1))

model.fit(x, y)
print(model.predict(21, return_std=True))

print(time.time()-t)