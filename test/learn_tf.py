import tensorflow as tf

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        print(v.name == "foo/bar/v:0")


with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
print(v1 == v)

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [2])
    v1 = tf.get_variable("v", [1])