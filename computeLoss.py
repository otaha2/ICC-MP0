import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf

# print(tf.__version__)

def function(session, loss):
	return session.run(loss)


if __name__ == "__main__":
	print("Compute Loss File")