import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import gzip

url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php'

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
	
	new_model = tf.keras.models.load_model('../Part4/keras_fashion_mnist_model.h5')

	for i in range(5):
		# Send post request for data
		predStr = ""
		data, testset_id = get_testset()
		data = data / 255.0
		data = np.reshape(data, (1000, 28, 28, 1))
		
		# Predict labels with model
		preds = new_model.predict_classes(data)

		# Construct pred string
		for pred in preds:
			predStr += str(pred)
		
		# Send results
		num_correct = send_results(predStr, testset_id)
		
		# Report Accuracy
		acc = int(num_correct)/1000.0
		print("Testset ID: ",testset_id , " Accuracy: ", acc)
		