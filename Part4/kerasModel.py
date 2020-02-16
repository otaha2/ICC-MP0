import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# Load data
	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
	
	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
		'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	train_images = train_images / 255.0
	test_images = test_images / 255.0

	input_shape = (28, 28, 1)

	# Build Model
	model = keras.Sequential([
		keras.layers.Conv2D(3, (5,5), strides=(1,1), padding="valid", activation="relu", input_shape=input_shape),
		keras.layers.MaxPooling2D((2,2)),
		keras.layers.Conv2D(3, (3,3), strides=(1,1), padding="valid", activation="relu"),
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

	# Fit Model on Training Data
	model.fit(train_images, train_labels, epochs=10)

	# Evaluate model on test data
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

	print('\nTest accuracy:', test_acc)