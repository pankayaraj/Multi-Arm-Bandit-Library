from Machine_learning.Bayesian_neural_network import BNN
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -(x**2 + 10*np.sin(x))


x = [-2, 3, 5]
y = [f(i) for i in x]


B = BNN(x, y, [20, 20])
B.train(2000, learning_rate=0.01, dropout_p=0.5)
itr = 200
x_ = []
a = np.linspace(-2, 5)
for i in a:
    for _ in range(itr):
        x_.append(i)

y_ = B.predict(x_)
print(len(y_))
mean = []
std = []
axis = []

for i in range(0, len(y_), itr):

    m = 0
    axis.append(x_[i])
    for j in range(itr):
        m += y_[i+j]
    m = m[0][0]/itr

    t = 0
    for j in range(itr):
        t += y_[i+j]**2

    v = t[0][0]/itr-m**2

    std.append(np.sqrt(abs(v)))
    mean.append(m)


print(x)
print(y)

print(axis)
print(mean)
print(std)

plt.plot(x, y,'y')
plt.plot(axis, mean, 'b')
#plt.plot(axis, std, 'r')
plt.show()