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
        self.bias = {}
        self.datatype = datatype
        self.dropout_p = 0.1

        self.init_var = 1/(self.output_dim+self.input_dim)

        self.weight['1'] = tf.Variable(tf.random_normal(shape=[self.architecture[0], self.input_dim],
                                                        mean=0, stddev=self.init_var, dtype=datatype), dtype=datatype)

        self.bias['1'] = tf.Variable(tf.random_normal(shape = [self.architecture[0], 1], dtype=self.datatype, mean=0
                                                     , stddev=self.init_var), dtype=self.datatype)


        i_ = 0
        for i in range(1, len(architecture)):
            i_ = i
            self.weight[str(i+1)] = tf.Variable(tf.random_normal(shape=[self.architecture[i], self.architecture[i-1]],
                                                                 mean=0, stddev=self.init_var, dtype=datatype), dtype= datatype)

            self.bias[str(i+1)] = tf.Variable(tf.random_normal(shape=[self.architecture[i], 1], dtype=self.datatype, mean=0
                                                          , stddev=self.init_var), dtype=self.datatype)


        self.weight[str(i_+2)] = tf.Variable(tf.random_normal(shape=[self.output_dim, self.architecture[-1]],
                                                             mean =0, stddev=self.init_var, dtype=datatype), dtype= datatype)

        self.bias[str(i_+2)] = tf.Variable(tf.random_normal(shape=[self.output_dim,1], dtype=self.datatype, mean=0
                                                             , stddev=self.init_var), dtype=self.datatype)


        print(self.weight)
        print(self.bias)


    def train(self, no_iterations = 100, dropout_p = 0.2, learning_rate= 0.01):

        self.dropout_p = dropout_p

        x = tf.placeholder(dtype=self.datatype, shape=[self.input_dim, 1], name="input")
        y = tf.placeholder(dtype=self.datatype, shape=[self.output_dim, 1], name="output")

        layer_output = {}
        layer_output['1'] = tf.nn.sigmoid(tf.add(tf.matmul(self.weight['1'], x), self.bias['1']))


        for i in range(len(self.architecture)):
            if i == len(self.architecture)-1:
                layer_output[str(i+2)] = tf.add(tf.matmul(self.weight[str(i+2)], layer_output[str(i+1)]), self.bias[str(i+2)])
            else:
                layer_output[str(i+2)] = tf.nn.sigmoid(tf.add(tf.matmul(self.weight[str(i+2)], tf.nn.dropout(layer_output[str(i+1)], keep_prob=dropout_p)),
                                                  self.bias[str(i+2)]))


        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        cost = tf.losses.mean_squared_error(y, layer_output[str(i+2)])

        train = optimizer.minimize(cost)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())


            for _ in range(no_iterations):
                for i in range(len(self.input)):
                   
                    sess.run(train, feed_dict={x: np.reshape(self.input[i], newshape=[self.input_dim, 1]),
                                                      y: np.reshape(self.output[i], newshape=[self.output_dim, 1])})

            weights = sess.run(self.weight)
            self.weight = weights

            biases = sess.run(self.bias)
            self.bias = biases

            '''
            print(sess.run(layer_output, feed_dict={x:np.reshape(self.input[0], newshape=[self.input_dim,1]),
                                         y:np.reshape(self.output[0], newshape=[self.output_dim,1])})['2'])

            print(sess.run(layer_output, feed_dict={x: np.reshape(self.input[1], newshape=[self.input_dim, 1]),
                                                    y: np.reshape(self.output[1], newshape=[self.output_dim, 1])})['2'])

            print(sess.run(layer_output, feed_dict={x: np.reshape(self.input[2], newshape=[self.input_dim, 1]),
                                                    y: np.reshape(self.output[2], newshape=[self.output_dim, 1])})['2'])
            '''

    def predict(self, X):


        inp = tf.placeholder(dtype=self.datatype, shape=[self.input_dim, 1])

        W = {}
        B = {}

        for i in range(len(self.architecture)+1):
            W[str(i+1)] = tf.constant(self.weight[str(i+1)], dtype=self.datatype)
            B[str(i+1)] = tf.constant(self.bias[str(i+1)], dtype=self.datatype)



        layer_output = {}
        layer_output['1'] = tf.nn.sigmoid(tf.add(tf.matmul(W['1'], inp), B['1']))


        for i in range(len(self.architecture)):
            if i == len(self.architecture)-1:
                layer_output[str(i+2)] = tf.add(tf.matmul(W[str(i+2)], layer_output[str(i+1)]), B[str(i+2)])
            else:
                layer_output[str(i+2)] = tf.nn.sigmoid(tf.add(tf.matmul(W[str(i+2)], tf.nn.dropout(layer_output[str(i+1)], keep_prob=self.dropout_p)),
                                                  B[str(i+2)]))


        prediction = [0]*len(X)
        with tf.Session() as sess:
            for i in range(len(X)):
                x = X[i]
                prediction[i] = sess.run(layer_output,
                         feed_dict={inp:np.reshape(x, newshape=[self.input_dim, 1])})[str(len(self.architecture)+1)]

        return prediction



B = BNN([1,2,3], [10,20,30], [10,10])
B.train(2000, learning_rate=0.01, dropout_p=0.5)
print(B.predict([1.5 for _ in range(10)]))