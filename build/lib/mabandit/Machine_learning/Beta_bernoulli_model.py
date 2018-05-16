import numpy as np
from scipy.stats import beta

class Beta_bernoulli():

    def __init__(self, a =1, b =1):
        self.a = a
        self.b = b

    def update(self, data):

        if data == 1:
            self.a += 1
        elif data == 0:
            self.b += 1

    def sample(self, size = 1):
        return np.random.beta(a = self.a, b = self.b, size=size)

    def PDF(self, x):
        return beta.pdf(x, self.a, self.b)

