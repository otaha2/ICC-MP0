import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import gzip

url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php'

def create_model():
	input_shape = (28, 28, 1)
	model = keras.Sequential([
		keras.layers.Conv2D(3, (5,5), strides=(1,1), padding="valid", activation="relu", input_shape=input_shape),
		keras.layers.MaxPooling2D((2,2)),
		keras.layers.Conv2D(3, (3,3), strides=(1,1), padding="same", activation="relu"),
		keras.layers.MaxPooling2D((2,2)),
		keras.layers.Flatten(),
		keras.layers.Dense(100, activation="relu"),
		keras.layers.Dense(50, activation="relu"),
		keras.layers.Dense(10, activation="softmax")
	])

	# Compile Model
	model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)
	
	return model

def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 784)
    return data

def get_testset():
    values = {'request': 'testdata', 'netid':'otaha2'}
    r = requests.post(url, data=values, allow_redirects=True)
    filename = r.url.split("/")[-1]
    testset_id = filename.split(".")[0].split("_")[-1]
    with open(filename, 'wb') as f: 
        f.write(r.content)
    return load_dataset(filename), testset_id
    
def send_results(pred, testset_id):
	values = {'request': 'verify', 'netid':'otaha2', 'testset_id': testset_id, 'prediction': pred}
	r = requests.post(url, data=values)
	return r.text

if __name__ == "__main__":
	predStr = ""
	
	# Create model & load weights
	model = create_model()
	model.load_weights('../Part4/checkpoints/end_checkpoint')

	saver = tf.train.Saver()
	# saver.restore(sess, '../Part4/low_level_cnn_model')

	imported_graph = tf.train.import_meta_graph('../Part4/low_level_cnn_model-250.meta')
	graph = tf.get_default_graph()
	X = graph.get_tensor_by_name('input_images:0')
	keep_prob = graph.get_tensor_by_name('dropout_placeholder:0')
	prediction = graph.get_tensor_by_name('prediction:0')



	# Send post request for data
	data, testset_id = get_testset()
	data = np.reshape(data, (1000, 28, 28, 1))
	
	# Predict labels with model
	preds_keras = model.predict(data)

	# with tf.Session() as sess:
	# 	imported_graph.restore(sess, '../Part4/low_level_cnn_model-250')
	# 	feed_dict = {X: data.reshape(1000, 784), keep_prob: 1.0}
	# 	preds = sess.run(prediction, feed_dict)
	# 	print(preds)

	
	
	# Construct pred string
	for pred in preds_keras:
		predStr += str(np.argmax(pred))
	
	# Send results
	num_correct = send_results(predStr, testset_id)
	
	# Report Accuracy
	acc = int(num_correct)/1000.0
	print("\nAccuracy: ", acc)
	