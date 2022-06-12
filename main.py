import tensorflow as tf
import cv2
import numpy
import matplotlib.pyplot as plt

Dataset = tf.keras.datasets.mnist
# importieren des Datasetes mnist
(x_train, y_train), (x_test, y_test) = Dataset.load_data()
# aufteilen des Datasets in x und y test

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#aufteilung des neuronalen Netzwerkes, in verschiedene layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
# algorithmus um das Neuronale Netzwerk lernen zu lassen
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10)

model.save("handwritten digits")




