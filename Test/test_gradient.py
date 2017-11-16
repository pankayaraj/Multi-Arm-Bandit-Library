import tensorflow as tf

x = tf.Variable(200, dtype=tf.float32)
opt = tf.train.GradientDescentOptimizer(5)
train = opt.minimize(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train)
        print(sess.run(x))