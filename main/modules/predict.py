import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tensorflow.python.keras.saving.saved_model.load import load

import os

cascade_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

HEIGHT, WIDTH = 64, 64
dim = (HEIGHT, WIDTH)


aux_model = load_model('modelos/fer2013_model.hdf5')
initial_model = load_model('modelos/fer2013_model.hdf5')
all_emotions = ['anger', 'disgust', 'fear',
                    'happiness', 'neutral', 'sadness', 'surprise']

def load_models(ini= 'modelos/fer2013_model.hdf5' ,other=''):
    initial_model = load_model(ini)
    if other:
        aux_model = load_model(other)

def get_prediction(imagen):
    """
    Return predictions of each image provided by main app.
    """
    try:
        imagen = cv2.resize(imagen, dim)
        print(imagen.shape)
        imagen_g = imagen.reshape(1, HEIGHT, WIDTH, 1)
        print(imagen_g.shape)
        prediction = initial_model.predict(imagen_g)
        print(prediction)
        print(prediction.argmax())
        return all_emotions[prediction.argmax()]

    except Exception as error:
        print(error)

def get_prediction_combined(imagen,initial,others):
    """
    Return predictions of each image provided by main app with 2 models.
    """    
    initial_emotions = [x for x in all_emotions if x in initial]
    other_emotions = [x for x in all_emotions if x not in initial_emotions]

    try:
        imagen = cv2.resize(imagen, dim)
        print(imagen.shape)
        imagen_g = imagen.reshape(1, HEIGHT, WIDTH, 1)
        print(imagen_g.shape)
        prediction = initial_model.predict(imagen_g)
        print(prediction)
        print(prediction.argmax())
        return all_emotions[prediction.argmax()]

    except Exception as error:
        print(error)
    pass