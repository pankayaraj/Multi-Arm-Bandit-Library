from Multi_arm_bandit_algorithms.Baysian_optimization.Explore_exploit_tradeoff import  Explore_exploit_tradeoff

X = [-5, -2, -1, 1, 2, 5]
Y = [-i**2 for i in X]

print(Explore_exploit_tradeoff(X, Y, lower_bound=-7, upper_bound=7, tradeoff_factor=100))

