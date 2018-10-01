#! /usr/bin/python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

mnist = input_data.read_data_sets('MNIST_data', one_hot= True)
# print(minst.train.images.shape)
# pip install h5py ==2.8.0rc1
lr =1e-3
batch_size = tf.placeholder(tf.int32)
input_size =28
timestep_size = 28
hidden_size = 256
layer_num = 2
class_num = 10
_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32,[None, class_num])
keep_prob = tf.placeholder(tf.float32)
X = tf.reshape(_X, [-1, 28, 28])
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units= hidden_size, forget_bias=1.0, state_is_tuple= True)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell= lstm_cell, input_keep_prob=1.0, output_keep_prob= keep_prob)
mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple= True)
init_state = mlstm_cell.zero_state(batch_size,dtype= tf.float32)
outputs = list()
state = init_state

with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1]

# W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev= 0.1), dtype= tf.float32)
# bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
# y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
#
# # 损失和评估函数
# cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
# train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#
# correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#
# sess.run(tf.global_variables_initializer())
# for i in range(2000):
#     _batch_size = 128
#     batch = mnist.train.next_batch(_batch_size)
#     if (i+1) % 200 == 0:
#         train_accuracy = sess.run(accuracy, feed_dict={
#             _X:batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
#         print ("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy))
#     sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})
#
# print("test accuracy %g"% sess.run(accuracy, feed_dict={
#     _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))
#
# import matplotlib.pyplot as plt
# print(mnist.train.labels[4])
#
# X3 = mnist.train.images[4]
# img3 = X3.reshape([28, 28])
# plt.imshow(img3, cmap='gray')
# plt.show()

