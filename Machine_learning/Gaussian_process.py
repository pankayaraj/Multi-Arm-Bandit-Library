import tensorflow as tf
import numpy as np
from Kernals.RBF import RBF

#Solving is done on numpy try to do it in tensorflow

class Gaussian_process():
    def __init__(self, datatype = tf.float64):
        self.y = None
        self.x = None
        self.zigma = None
        self.mean = None
        self.no_points = None
        self.K = None
        self.variance_weight = 1
        self.datatype = datatype


    def fit_noiseless(self, x, y, mean =0, zigma = 1, variance_weight = 1):
        self.x = x
        self.y = tf.constant([[i] for i in y], dtype=self.datatype)
        self.zigma = zigma
        self.variance_weight = variance_weight
        self.mean = mean
        self.no_points = len(y)
        self.K = tf.constant([[RBF(x[i], x[j]) for j in range(self.no_points)] for i in range(self.no_points)], dtype=self.datatype)

    def fit_noisy(self, x, y, noise, mean =0, zigma = 1, variance_weight = 1):
        self.x = x
        self.y = tf.constant([[i] for i in y], dtype=self.datatype)
        self.zigma = zigma
        self.variance_weight = variance_weight
        self.mean = mean
        self.no_points = len(y)
        self.K = tf.constant([[RBF(x[i], x[j]) for j in range(self.no_points)] for i in range(self.no_points)], dtype=self.datatype) + noise*tf.eye(self.no_points, self.datatype)

    def predict(self,x):

        k = tf.constant([[RBF(x, self.x[i])] for i in range(self.no_points)], dtype=self.datatype)


        try:
            L = tf.cholesky(self.K)
            alpha = tf.matrix_triangular_solve(L, self.y, lower=True)
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
            K_inverse = tf.matrix_inverse(self.K)
            alpha = tf.matmul(K_inverse, self.y)
            beta = tf.matmul(k, alpha, transpose_a=True)

            gamma = tf.matmul(K_inverse, k)
            v = tf.matmul(k, gamma, transpose_a=True)

            with tf.Session() as sess:
                mean = self.mean + sess.run(beta)
                variance = self.variance_weight - sess.run(v)

            return mean, variance

