import tkinter as tk
from PIL import ImageTk, Image  # pip3 install Pillow
from tkinter import filedialog

import engine.torch as tengine


def upload():  # AQUI SE SUBE LA IMAGEN
    filename = filedialog.askopenfilename(title='open', filetypes=[("Images", ".jpg")])
    img = Image.open(filename)
    ph = ImageTk.PhotoImage(img)
    print(filename)
    # do prediction here
    # ...
    img, has_covid = trunner.predict(img, filename)

    tk_img = img.resize((256, 256), Image.ANTIALIAS)
    tk_img = ImageTk.PhotoImage(tk_img)
    panel = tk.Label(mainWindow, image=tk_img)
    panel.image = tk_img
    panel.pack()
    panel.place(x=400, y=50)


def process_img():  # AQUI SE LLAMA AL MODELO PARA ANALIZAR LA IMAGEN
    covidPositive = False
    textResult = "El individuo no presenta COVID-19"

    if covidPositive:
        textResult = "El individuo si presenta COVID-19"

    result = tk.Label(mainWindow, text=textResult)
    result.pack(anchor=tk.NW)
    result.config(fg="red", bg="#c3d6ff", font=("Arial", 14))
    result.place(x=25, y=150)


trunner = tengine.TorchEngine()
mainWindow = tk.Tk()
mainWindow.title("Deteccion de COVID-19")
mainWindow.geometry("700x400")
mainWindow.config(bg="#c3d6ff")

title = tk.Label(mainWindow, text="Deteccion de COVID-19")
title.pack(anchor=tk.NW)
title.config(fg="red", bg="#c3d6ff", font=("Arial", 22))
title.place(x=25)

uploadButton = tk.Button(mainWindow, text="Subir rayos x...", height=2, width=20, command=upload)
uploadButton.pack(anchor=tk.NW)
uploadButton.config(bg="#c0c0c0", font=("Arial", 9))
uploadButton.place(x=25, y=50)

processButton = tk.Button(mainWindow, text="Procesar", height=2, width=20, command=process_img)
processButton.pack(anchor=tk.NW)
processButton.config(bg="#c0c0c0", font=("Arial", 9))
processButton.place(x=200, y=50)

mainWindow.mainloop()
