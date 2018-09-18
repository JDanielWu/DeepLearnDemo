import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import tensorflow.contrib.slim as slim

mnist = input_data.read_data_sets('./ministdata', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# cast x to 3D
x_image = tf.reshape(x, [-1, 28, 28, 1], name='input')  # shape of x is [N,28,28,1]

# conv layer1
net = slim.conv2d(x_image, 32, [5, 5], scope='conv1')  # shape of net is [N,28,28,32]
net = slim.max_pool2d(net, [2, 2], scope='pool1')  # shape of net is [N,14,14,32]

# conv layer2
net = slim.conv2d(net, 64, [5, 5], scope='conv2')  # shape of net is [N,14,14,64]
net = slim.max_pool2d(net, [2, 2], scope='pool2')  # shape of net is [N,7,7,64]

# reshape for full connection
net = tf.reshape(net, [-1, 7 * 7 * 64])  # [N,7*7*64]

# fc1
net = slim.fully_connected(net, 128, scope='fc1')  # shape of net is [N,1024]

# dropout layer
# keep_prob = tf.placeholder('float')			#As toco not support dropout now, so I delete it.
# net = tf.nn.dropout(net, keep_prob)
# fc2
net = slim.fully_connected(net, 10, scope='fc2')  # [N,10]
# softmax
y = tf.nn.softmax(net, name='output')  # [N,10]

cross_entropy = -tf.reduce_sum(tf.multiply(y_, tf.log(y)))  # y and _y have same shape.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))  # shape of correct_prediction is [N]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(6000):
        batch = mnist.train.next_batch(50)
        if i % 50 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]})
            print('step %d,training accuracy  %g !!!!!!!' % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    batch = mnist.train.next_batch(50)
    total_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]})
    print('test_accuracy  %s!!!!!!!' % (total_accuracy))
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names=["output"])

    tf.train.write_graph(frozen_graph_def, './', 'minist.pb',as_text=False)
