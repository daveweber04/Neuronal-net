from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui as wg
from PIL import ImageGrab,Image
import numpy as np
model = load_model("handwritten.model")
def prediction_of_digits(img):
    #bild umformatieren auf 28*28 Pixel
    img = img.resize((28,28))
    #umwandeln des rgb ins grayscale
    img = img.convert('L')
    img = np.array(img)
    #reformatieren des inputs (um sich dem modell anzupassen) des Modells und normalisieren
    img = img.reshape(1,28,28)
    img = img/255.0
    #vorhersagen der klasse
    res = model.predict([img])[0]
    return np.argmax(res), max(res)
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Elemente kreiren
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command =         self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Struktur des Grids
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #Self canvas kreieren
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = wg.GetWindowRect(HWND) # koordinaten des Canvas
        im = ImageGrab.grab(rect)
        digit, acc = prediction_of_digits(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
app = App()
mainloop()