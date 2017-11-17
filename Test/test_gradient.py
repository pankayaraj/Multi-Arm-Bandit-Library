from scipy.optimize import minimize

def f(x):
    def g(x):
        print(x)

    g(20)

f(2)



