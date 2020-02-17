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

	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k=2)

	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], padding='SAME')
	conv2 = maxpool2d(conv2, k=2, padding='SAME')

	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# fc2 = tf.nn.dropout(fc1, dropout)

	# fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
	fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
	fc2 = tf.nn.relu(fc2)

	# fc2 = tf.nn.dropout(fc2, dropout)

	out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

	return out

if __name__ == "__main__":

	mnist = input_data.read_data_sets('data/fasion', one_hot=True)

	learning_rate = 0.002
	num_steps = 500
	batch_size = 128
	display_step = 10

	num_input = 784
	num_classes = 10
	dropout = 0.75
	epochs = 5

	loss_val = []

	X = tf.placeholder(tf.float32, [None, num_input])
	Y = tf.placeholder(tf.float32, [None, num_classes])
	keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

	weights = {
		'wc1': tf.Variable(tf.random_normal([5, 5, 1, 3])),
		'wc2': tf.Variable(tf.random_normal([3, 3, 3, 3])),
		'wd1': tf.Variable(tf.random_normal([6*6*3, 100])),
		'wd2': tf.Variable(tf.random_normal([100, 50])),
		'out': tf.Variable(tf.random_normal([50, num_classes]))
	}

	biases = {
		'bc1': tf.Variable(tf.random_normal([3])),
		'bc2': tf.Variable(tf.random_normal([3])),
		'bd1': tf.Variable(tf.random_normal([100])),
		'bd2': tf.Variable(tf.random_normal([50])),
		'out': tf.Variable(tf.random_normal([num_classes]))
	}

	# Construct model
	logits = conv_net(X, weights, biases, keep_prob)
	prediction = tf.nn.softmax(logits)

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

	#Start training
	with tf.compat.v1.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			print("\nEpoch #", epoch)
			for step in range(1, num_steps+1):
				batch_x, batch_y = mnist.train.next_batch(batch_size)

				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
				if step % display_step == 0 or step == 1:
					loss, acc = sess.run([loss_op, accuracy], 
						feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})

					print("Step " + str(step) + ", Minibatch Loss= " + \
	                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
	                  "{:.3f}".format(acc))
			loss_val.append(loss)

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




