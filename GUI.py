from tkinter import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import load_model
from matplotlib import image as pltImage
from skimage.transform import resize
from PIL import Image, ImageTk, ImageFilter
import numpy as np


def load_MNIST_model(filepath):
    return load_model(filepath)


class MNIST(object):
    def __init__(self):
        self.prediction_txt = 'Prediction:'
        self.root = Tk()
        self.root.title('Number Prediction')
        self.root.resizable(False, False)
        self.CANVAS_WIDTH = 280
        self.CANVAS_HEIGHT = 280
        self.model = load_MNIST_model("models/SGD.keras")  # Laden des Modells

        # Erstellen des 'Clear' Buttons
        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=0, column=0)

        # Vorhersage
        self.predicted_num_label = Label(self.root, text=self.prediction_txt, anchor='w')
        self.predicted_num_label.grid(row=0, column=1, columnspan=4, sticky='nsew')

        self.canvas = Canvas(self.root, bg='white', width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
        self.canvas.grid(row=1, columnspan=6)
        self.setup()
        self.root.mainloop()

    def canvas_to_ndarray(self):
        try:
            x = self.root.winfo_rootx() + self.canvas.winfo_x()
            y = self.root.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()

            # Verwende ImageGrab.grab, um einen Screenshot des Canvas zu machen.
            # Stelle sicher, dass du die richtigen Koordinaten angibst.
            # bbox ist ein Tupel (x0, y0, x1, y1)
            # korrekte verwendung von ImageGrab.grab, um fehler zu vermeiden
            img = ImageGrab.grab(bbox=(x, y, x1, y1))

            # Konvertiere das Bild in Graustufen.
            img = img.convert("L")
            img = img.resize((28, 28), Image.LANCZOS).filter(ImageFilter.SHARPEN)
            arr = np.array(img)
            arr = (255 - arr) / 255.0
            return arr

        except Exception as e:
            print(f"Fehler bei der Bildverarbeitung: {e}")
            return None  # Oder wirf eine Exception, wenn du den Fehler nicht hier behandeln willst.

    def clear(self):
        self.canvas.delete('all')
        self.predicted_num_label['text'] = self.prediction_txt
        self.old_x = None
        self.old_y = None

    def reset(self, event):
        self.old_x = None
        self.old_y = None
        arr = self.canvas_to_ndarray()
        if arr is not None: # stelle sicher das arr kein 'NoneType' ist
            pred = self.model.predict(arr.reshape(-1, 28, 28, 1))
            prediction = np.argmax(pred, axis=1)[0]
            percent = np.sort(pred, kind='mergesort')[0, -1] * 100
            txt = '{} {} ({}%)'.format(self.prediction_txt.strip(), prediction, round(percent, 2))
            self.predicted_num_label['text'] = txt
        else:
            self.predicted_num_label['text'] = "Fehler bei der Vorhersage"


    def setup(self):
        self.old_x = None
        self.old_y = None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        self.line_width = 20
        paint_color = 'black'
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.line_width, fill=paint_color,
                                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
from PIL import ImageGrab

if __name__ == "__main__":
    MNIST()
