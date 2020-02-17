import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

def conv2d(x, W, b, strides=1, padding='VALID'):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2, padding='VALID'):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

def conv_net(x, weights, biases, dropout):
	x = tf.reshape(x, shape=[-1, 28, 28, 1])

	conv1 = conv2d(x, weights['wc1'], biases['bc1'], padding='SAME')
	conv1 = maxpool2d(conv1, k=2, padding='SAME')

	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], padding='SAME')
	conv2 = maxpool2d(conv2, k=2, padding='SAME')

	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, dropout)

	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

	return out

if __name__ == "__main__":

	mnist = input_data.read_data_sets('data/fasion', one_hot=True)

	learning_rate = 0.002
	num_steps = 250
	batch_size = 256
	display_step = 10

	num_input = 784
	num_classes = 10
	dropout = 0.75
	epochs = 3

	loss_val = []

	X = tf.placeholder(tf.float32, [None, num_input], name='input_images')
	Y = tf.placeholder(tf.float32, [None, num_classes], name='input_labels')
	keep_prob = tf.placeholder(tf.float32, name='dropout_placeholder') # dropout (keep probability)

	weights = {
		'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
		'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
		'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
		# 'wd2': tf.Variable(tf.random_normal([100, 50])),
		'out': tf.Variable(tf.random_normal([1024, num_classes]))
	}

	biases = {
		'bc1': tf.Variable(tf.random_normal([32])),
		'bc2': tf.Variable(tf.random_normal([64])),
		'bd1': tf.Variable(tf.random_normal([1024])),
		# 'bd2': tf.Variable(tf.random_normal([50])),
		'out': tf.Variable(tf.random_normal([num_classes]))
	}

	# Construct model
	logits = conv_net(X, weights, biases, keep_prob)
	prediction = tf.nn.softmax(logits, name='prediction')

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=logits, labels=Y))
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	#Initialize variables
	init = tf.compat.v1.global_variables_initializer()

	saver = tf.train.Saver()

	#Start training
	with tf.compat.v1.Session() as sess:
		sess.run(init)

		# for epoch in range(epochs):
			# print("\nEpoch #", epoch)
		for step in range(1, num_steps+1):
			batch_x, batch_y = mnist.train.next_batch(batch_size)

			sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
			if step % display_step == 0 or step == 1:
				loss, acc = sess.run([loss_op, accuracy], 
					feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
				# if step == 1:
				# 	loss_val.append(loss)

				print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
			loss_val.append(loss)

		saved_path = saver.save(sess, './low_level_cnn_model', global_step=step)

		print("Optimization Finished!")

		print("Testing Accuracy:", \
			sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
										Y: mnist.test.labels[:256],
										keep_prob: 1.0})
			)

		

	plt.plot(loss_val)
	plt.title('Low Level API Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.show()




