from mabandit.Machine_learning.Beta_bernoulli_model import Beta_bernoulli


class Thompson_sampler():

    def __init__(self, no_arms):
        self.no_arms = no_arms
        self.arms_prior_p = [Beta_bernoulli() for i in range(no_arms)]


    def feed(self, arm_no, data):
        self.arms_prior_p[arm_no-1].update(data)

    def next_arm(self):
        reward_p = [self.arms_prior_p[i].sample() for i in range(self.no_arms)]

        index = 0
        max = reward_p[0]
        for i in range(self.no_arms):
            if reward_p[i] > max:
                max = reward_p[i]
                index = i

        next_arm = index+1

        return next_arm

    def return_arm_paramaeter(self, arm_no):
        return self.arms_prior_p[arm_no-1].sample()

    def return_arm_paramaeter_pdf(self, arm_no, x):
        return self.arms_prior_p[arm_no-1].PDF(x)
