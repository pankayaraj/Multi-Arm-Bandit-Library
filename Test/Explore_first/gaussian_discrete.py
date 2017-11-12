import numpy as np
#import matplotlib.pyplot as plt
from Multi_arm_bandit_algorithms.Non_adaptive_algoritms.Explore_first import Explore_first

arms = np.linspace(-1, 1, 40)
rewards = []
print(arms)
E = Explore_first(1000, arms)






