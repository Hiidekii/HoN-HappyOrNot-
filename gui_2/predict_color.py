import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')

HEIGHT, WIDTH = 75,75
dim=(HEIGHT, WIDTH)

cats = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

new_model = load_model("faces_iception_2_best_model20211112.h5")
new_model.summary()

def get_face(img):
    try:
        print(face_cascade.detectMultiScale(img, 1.3, 5))
        x, y, w, h = face_cascade.detectMultiScale(img, 1.3, 5)
        return img[y:y+h, x:x+w]
    except:
        print("no estoy decetando rostro")

i=0

def get_prediction(imagen):
    try:
        # imagen = get_face(img)
        #imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = cv2.resize(imagen, dim)
        print(imagen.shape)
        #imagen_g =imagen.reshape(1,48,48) #IMPRESCINDIBLE PARA MODELO GRAY
        imagen_g =imagen.reshape(1,75,75,3)
        print(imagen_g.shape)
        #cv2.imwrite('pruebas/waka.jpg', imagen)
        prediction = new_model.predict(imagen_g)
        return cats[prediction.argmax()]
    except Exception as error:
        print(error)
