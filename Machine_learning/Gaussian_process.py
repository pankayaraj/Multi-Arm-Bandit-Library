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

    def predict(self,x):
        K = tf.constant(self.K)
        k = tf.constant([[RBF(x, self.x[i])] for i in range(self.no_points)])
        y = tf.constant([[i] for i in self.y], dtype=tf.float32)

        try:
            L = tf.cholesky(K)
            alpha = tf.matrix_triangular_solve(L, y, lower=True)
            beta  = tf.matrix_triangular_solve(tf.transpose(L), alpha, lower=False)

            m = self.mean + tf.matmul(k, beta, transpose_a=True)

            gamma = tf.matrix_triangular_solve(L, k)
            v = tf.matmul(gamma, gamma, transpose_a=True)

            with tf.Session() as sess:
                mean = sess.run(m)
                variance = self.variance_weight - sess.run(v)
            return mean, variance
        except:
            print("e")
            K_inverse = tf.matrix_inverse(K)
            alpha = tf.matmul(K_inverse, y)
            beta = tf.matmul(k, alpha, transpose_a=True)

            gamma = tf.matmul(K_inverse, k)
            v = tf.matmul(k, gamma, transpose_a=True)

            with tf.Session() as sess:
                mean = self.mean + sess.run(beta)
                variance = self.variance_weight - sess.run(v)

            return mean, variance

