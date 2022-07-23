import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui as wg
from PIL import ImageGrab,Image

def plot_history(history):
    #defienieren des Plotes
    fig,axs=plt.subplots(2)
    axs[0].plot(history.history["accuracy"],label="train accuracy")
    #genauigkeit des neuronalen Netzwerkes
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    #genauigkeit des Netzwerkes bei der überprüfung
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    #beschriftung hinzufügen.
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"],label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")


    plt.show()



Dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = Dataset.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
#definieren der trainigsdaten
x_test = tf.keras.utils.normalize(x_test, axis=1)
#defeinieren der überprüfungsdaten

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
#definieren der vercshiedenen Layer des Netzwerkes

model.compile(optimizer="SGD", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#konfigurieren für das Training

history=model.fit(x_train, y_train,validation_data=(x_test,y_test),batch_size=32,epochs=60)
model.save("SGD.model")
#abspeichern des Models
plot_history(history)
#plotten des models




