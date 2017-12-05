import tensorflow as tf
import numpy as np

class BNN():

    def __init__(self, X, Y, architecture = [10], datatype = tf.float64):
        try:
            self.input_dim = len(X[0])
        except:
            self.input_dim = 1

        try:
            self.output_dim = len(Y[0])
        except:
            self.output_dim = 1

        self.input = X
        self.output = Y
        self.architecture = architecture
        self.weight = {}
        self.datatype = datatype

        self.init_var = 1/(self.output_dim+self.input_dim)

        self.weight['1'] = tf.Variable(tf.random_normal(shape=[self.architecture[0], self.input_dim],
                                                        mean=0, stddev=self.init_var, dtype=datatype), dtype=datatype)

        for i in range(1, len(architecture)):
            self.weight[str(i+1)] = tf.Variable(tf.random_normal(shape=[self.architecture[i], self.architecture[i-1]],
                                                                 mean=0, stddev=self.init_var, dtype=datatype), dtype= datatype)

        self.weight[str(i+2)] = tf.Variable(tf.random_normal(shape=[self.output_dim, self.architecture[-1]],
                                                             mean =0, stddev=self.init_var, dtype=datatype), dtype= datatype)

    def train(self, no_iterations = 100, dropout_p = 0.2, learning_rate= 0.1):


        X_ = {}



        X_['1'] = tf.placeholder(dtype=self.datatype, shape=[1, self.input_dim], name='input')

        for i in range(len(self.architecture)+1):

            X_[str(i+2)] = tf.nn.relu(tf.matmul(tf.nn.dropout(X_[str(i+1)], keep_prob=dropout_p), self.weight[str(i+1)], transpose_b=True))

        print(X_)
        print(self.weight)
        print(np.shape(np.reshape(self.output[0], newshape= [1,self.output_dim])))

        Y = tf.placeholder(dtype=self.datatype, shape = [1,self.output_dim], name='output')
        print(Y)

        cost = tf.reduce_mean(tf.losses.mean_squared_error(Y, X_[str(i+2)]))
        optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            for i in range(no_iterations):
                for j in range(len(self.input)):

                    sess.run(optimize, feed_dict={X_['1']:np.reshape(np.array(self.input[j], dtype='d'), newshape=[1, self.input_dim]),
                                                  Y:np.reshape(self.output[j], newshape= [1,self.output_dim])})

                    print(sess.run(X_))
                    print(sess.run(Y))



b = BNN([[1.0,1.0]], [1], [2,3])
b.train(no_iterations=3)
feed_dict={x:[1,1]}))



