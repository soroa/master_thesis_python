import tensorflow as tf

# a = tf.constant(5, name="input_a")
# b = tf.constant(4, name="input_b")
# c = tf.multiply(a,b, name="multiply_c")
#
# sess = tf.Session()
# output = sess.run(c)
# writer = tf.summary.FileWriter('./my_graph', sess.graph)
# writer.close()
# sess.close()


W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32, name="x")
linear_model = W * x + b
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
