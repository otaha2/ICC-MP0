import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf

print(tf.__version__)

def function(session, optimizer):
	session.run(optimizer)

if __name__ == "__main__":
	print("TrainStep File")