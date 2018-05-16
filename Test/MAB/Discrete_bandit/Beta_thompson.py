from  Multi_arm_bandit_algorithms.Beta_models.Thompson_sampling import Thompson_sampler
import numpy as np
import matplotlib.pyplot as plt

T = Thompson_sampler(5)
iter = 150

arms = [0.1, 0.5, 0.7, 0.2, 0.8]
error = [[] for i in range(5)]
total_error =[0 for i in range(150)]

for _ in range(iter):
    x = T.next_arm()
    T.feed(x, np.random.binomial(1, arms[x-1]))


    for i in range(5):
        max = 0
        index = 0
        b = np.linspace(0,1,100)
        for a in b:
            y = T.return_arm_paramaeter_pdf(i+1, a)
            if y > max:
                max = y
                index= a
        error[i].append(abs(arms[i]-index))
        total_error[_] += abs(arms[i]-index)

n = [i for i in range(1,iter+1)]
fig = plt.figure()
plt.plot(n, total_error)
plt.xlabel("No_iterations")
plt.ylabel("Error")
plt.savefig("total_error")

#fig, arr = plt.subplots(5, sharey=True)
for i in range(5):
    s = "Theta = " + str(arms[i])
    print(s)
    fig = plt.figure()
    #arr[i].plot(n, error[i])
    #arr[i].set_title('Theta = ' + str(arms[i]))

    plt.plot(n, error[i])

    plt.xlabel("No_iterations")
    plt.ylabel("Error")
    fig.savefig(str(i+1))

