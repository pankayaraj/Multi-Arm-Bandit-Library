# Radial bias kernel

import tensorflow as tf
import numpy as np

#This was a huge bottelneck see why
''''
def RBF(a, b, zigma = 1, variance_weight = 1, datatype = "float64"):
    v1 = tf.Variable(a, dtype=tf.float64)
    v2 = tf.Variable(b, dtype=tf.float64)

    euclidian_dist = tf.reduce_sum(tf.square(tf.subtract(v1,v2)))
    ans = tf.exp(-0.5*euclidian_dist/zigma)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       answer = variance_weight*sess.run(ans)

    return answer
'''
def RBF(a, b, zigma=1, variance_weight =1):
    v1 = np.array(a)
    v2 = np.array(b)

    euclidian_distance = np.sum(np.square(np.subtract(v1,v2)))
    ans = np.exp(-0.5 *euclidian_distance/zigma)

    return ans