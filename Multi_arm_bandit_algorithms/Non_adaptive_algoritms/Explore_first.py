import numpy as np
import tensorflow as tf
from Data_structures.Priority_queue import Priority_queue

class Explore_first():

    def __init__(self, time_horizon, arms):
        self.state = "Explore"
        self.time_horizon = time_horizon
        self.arms = arms
        self.no_arms = len(arms)

        self.no_of_explorations = np.floor(((self.time_horizon/self.no_arms)**(2/3))*(np.log(time_horizon)**(1/3)))

        self.arms_tracker = [0]*self.no_arms
        self.current_arm_index = 0
        self.previous_arm_index = 0
        self.average_rewards = [[0, i] for i in arms]
        self.maximum_reward_arm = None
        self.arms_counter = {}
        for i in arms:
            self.arms_counter[i] = 0
        self.total_reward = 0


    def explore_and_exploit(self):
        if self.arms_tracker[-1] < self.no_of_explorations:
            return self.explore()
        else:
            return self.exploit()

    def explore(self):
        temp = self.current_arm_index
        self.current_arm_index = divmod(self.current_arm_index+1, self.no_arms)[1]
        self.arms_tracker[temp] += 1
        self.arms_counter[self.arms[temp]] += 1
        self.previous_arm_index = temp
        return self.arms[temp]

    def exploit(self):

        if self.maximum_reward_arm is None:
            self.state = "Exploit"
            self.average_rewards.sort()
            self.maximum_reward_arm = self.average_rewards[-1][1]
            self.arms_counter[self.maximum_reward_arm] += 1
            self.previous_arm_index = 0
            self.current_arm_index = 0
            return self.maximum_reward_arm
        else:
            self.arms_counter[self.maximum_reward_arm] += 1
            return self.maximum_reward_arm

    def feed_reward(self, reward):

        if self.state == "Explore":
            current_arm_index_tem = self.previous_arm_index
        else:
            current_arm_index_tem = -1
        self.average_rewards[current_arm_index_tem][0] = (self.average_rewards[current_arm_index_tem][0]*(self.arms_tracker[self.previous_arm_index]-1) + reward)/self.arms_tracker[self.previous_arm_index]
        self.total_reward += reward

    def get_description(self):
        print("Total reward = " + str(self.total_reward))
        print(self.average_rewards)
        print(self.arms_counter)

    def finalize(self):
        print("Total reward = " + str(self.total_reward))
        print("Optimal arm = " + str(self.maximum_reward_arm))
        print(self.average_rewards) #This is the average rewards at the end not after exploration phase
        print(self.arms_counter)


#TEST
'''

e = Explore_first(10, ['a', 'b', 'c', 'd'])
for _ in range(10):
    r = np.random.random()
    arm = e.explore_and_exploit()
    print(arm + " " + str(r))
    e.feed_reward(r)
    print(e.average_rewards)

e.finalize()

'''