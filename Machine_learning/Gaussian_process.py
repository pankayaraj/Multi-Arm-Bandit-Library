import tensorflow as tf
import numpy as np
from Kernals.RBF import RBF

#Solving is done on numpy try to do it in tensorflow

class Gaussian_process():
    def __init__(self):
        self.y = None
        self.x = None
        self.zigma = None
        self.mean = None
        self.no_points = None
        self.K = None
        self.variance_weight = 1


    def fit_noiseless(self, x, y, mean =0, zigma = 1, variance_weight = 1):
        self.x = x
        self.y = y
        self.zigma = zigma
        self.variance_weight = variance_weight
        self.mean = mean
        self.no_points = len(y)
        self.K = [[RBF(x[i], x[j]) for j in range(self.no_points)] for i in range(self.no_points)]

    def predict(self, x):
        K = tf.Variable(self.K, dtype=tf.float64)
        K_inverse = tf.linalg.inv(K)
        k_ = [RBF(x, self.x[i], zigma=self.zigma, variance_weight=self.variance_weight) for i in range(self.no_points)]
        k = tf.Variable(k_, dtype=tf.float64)
        y = tf.Variable(self.y, dtype=tf.float64)
        try:
            L = tf.cholesky(K)
            L_transpose = tf.transpose(L)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                L_trans = sess.run(L_transpose)
                L_ = sess.run(L)


            Beta = np.linalg.solve(L_, self.y)
            Alpha = tf.Variable(np.linalg.solve(L_trans, Beta))
            m = tf.einsum('i,i->',k, Alpha)
            gamma = tf.Variable(np.linalg.solve(L_, k_))
            v = self.variance_weight - tf.einsum('i,i->',gamma, gamma)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                mean = sess.run(m)
                variance = sess.run(v)

            return mean, variance

        except:
            print("e")
            alpha = tf.einsum('nm,m->n', K_inverse, y)
            m = tf.einsum('i,i->', k, alpha)

            v = self.variance_weight - tf.einsum('i,i->', tf.einsum('n,nm->m', k, K_inverse), k)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                mean = sess.run(m)
                variance = sess.run(v)

            return mean, variance

