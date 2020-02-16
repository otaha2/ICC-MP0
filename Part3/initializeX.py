import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import numpy as np

# print(tf.__version__)

def function(shape):
	arr = np.random.rand(shape[0], shape[1])
	var_arr = tf.get_variable("tf_var_init_from_np", initializer=arr)
	return tf.cast(var_arr, tf.float32)


if __name__ == "__main__":
	print(np.random.rand(4, 1))
