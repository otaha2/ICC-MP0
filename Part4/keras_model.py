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
		keras.layers.Conv2D(3, (5,5), strides=(1,1), padding="valid", activation="relu", input_shape=input_shape
			), # kernel_initializer=tf.keras.initializers.GlorotNormal
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
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	# Fit Model on Training Data
	history = model.fit(train_images, train_labels, epochs=10) #, validation_split=0.33

	model.save('keras_fashion_mnist_model.h5')

	# Evaluate model on test data
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

	print('\nTest accuracy:', test_acc)

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.title('Keras Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.show()

