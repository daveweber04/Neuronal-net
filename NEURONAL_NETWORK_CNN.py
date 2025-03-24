import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # Wird nicht direkt verwendet, könnte aber für spätere Änderungen nützlich sein
import matplotlib.pyplot as plt

def plot_history(history):
    """
    Stellt den Trainingsverlauf des Modells grafisch dar.
    Zeigt die Genauigkeit (Accuracy) und den Verlust (Loss) für Trainings- und Validierungsdaten
    über die Epochen hinweg an.
    """
    plt.figure(figsize=(12, 4))  # Größe des Diagramms festlegen

    # Subplot für die Genauigkeit
    plt.subplot(1, 2, 1)  # 1 Zeile, 2 Spalten, erstes Diagramm
    plt.plot(history.history['accuracy'], label='Training Accuracy')  # Trainingsgenauigkeit plotten
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Validierungsgenauigkeit plotten
    plt.xlabel('Epoch')  # X-Achsenbeschriftung
    plt.ylabel('Accuracy')  # Y-Achsenbeschriftung
    plt.title('Accuracy over Epochs')  # Titel des Diagramms
    plt.legend()  # Legende anzeigen
    plt.grid(True)  # Gitterlinien anzeigen

    # Subplot für den Verlust
    plt.subplot(1, 2, 2)  # 1 Zeile, 2 Spalten, zweites Diagramm
    plt.plot(history.history['loss'], label='Training Loss')  # Trainingsverlust plotten
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Validierungsverlust plotten
    plt.xlabel('Epoch')  # X-Achsenbeschriftung
    plt.ylabel('Loss')  # Y-Achsenbeschriftung
    plt.title('Loss over Epochs')  # Titel des Diagramms
    plt.legend()  # Legende anzeigen
    plt.grid(True)  # Gitterlinien anzeigen

    plt.tight_layout()  # Verhindert Überlappungen der Diagramme
    plt.show()  # Diagramm anzeigen


# Laden des MNIST-Datensatzes (handgeschriebene Ziffern)
dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Vorbereiten der Bilddaten
# Reshape:  (Anzahl Bilder, Höhe, Breite, Kanäle)  ->  Graustufenbilder haben 1 Kanal
x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype("float32") / 255  # Normalisieren der Pixelwerte (0-255 -> 0-1)
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype("float32") / 255


# Definieren des Convolutional Neural Network (CNN) Modells
model = tf.keras.models.Sequential()  # Ein sequentielles Modell (Schichten linear hintereinander)
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))  # Convolutional Layer 1
model.add(tf.keras.layers.MaxPooling2D((2, 2)))  # Max-Pooling Layer 1
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))  # Convolutional Layer 2
model.add(tf.keras.layers.MaxPooling2D((2, 2)))  # Max-Pooling Layer 2
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))  # Convolutional Layer 3
model.add(tf.keras.layers.Flatten())  # Macht aus dem 2D-Output einen 1D-Vektor
model.add(tf.keras.layers.Dense(64, activation="relu"))  # Fully Connected Layer (Dense Layer)
model.add(tf.keras.layers.Dense(10, activation="softmax"))  # Output Layer (10 Neuronen für 10 Ziffern, Softmax für Wahrscheinlichkeiten)

# Kompilieren des Modells
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#   optimizer:  Wie das Modell lernt (z.B. "rmsprop", "adam")
#   loss:       Wie der Fehler berechnet wird ("sparse_categorical_crossentropy" für Integer-Labels)
#   metrics:    Welche Metriken während des Trainings überwacht werden (hier: "accuracy")

# Trainieren des Modells
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=4)
#   x_train: Trainingsbilder
#   y_train: Trainingslabels (Ziffern 0-9)
#   validation_data: Daten zur Überprüfung der Leistung während des Trainings
#   batch_size: Anzahl der Bilder, die pro Gradientenaktualisierung verwendet werden
#   epochs: Anzahl der Durchläufe durch den gesamten Trainingsdatensatz

# Speichern des trainierten Modells
model.save("models/CNN.keras")

# Anzeigen des Trainingsverlaufs (Genauigkeit und Verlust)
plot_history(history)