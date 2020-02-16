import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf

print(tf.__version__)

def function(loss, lr):
	opt = tf.compat.v1.train.AdamOptimizer(lr, name='AdamOptimizer')
	opt_op = opt.minimize(loss)
	return opt_op

if __name__ == "__main__":
	print("Optimizer File")