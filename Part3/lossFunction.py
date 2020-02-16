import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import numpy as np

# print(tf.__version__)

def function(a, X, b, y):
	# (a(X^t*X) + b^t*X - y) ** 2
	
	transposeX_op = tf.transpose(X, name='transposeX')
	transposeb_op = tf.transpose(b, name='transposeb')

	matmulXX_op = tf.linalg.matmul(transposeX_op, X, name='matmulXX')
	matmulbX_op = tf.linalg.matmul(transposeb_op, X, name='matmulbX')

	mul_op = tf.math.multiply(a, matmulXX_op, name='mulaXX')
	add_op = tf.math.add(mul_op, matmulbX_op, name='add')
	sub_op = tf.math.subtract(add_op, y, name='sub')
	pow_op = tf.math.pow(sub_op, tf.constant(2, dtype=tf.float32), name='pow')

	return pow_op

if __name__ == "__main__":
	print("Create Loss File")