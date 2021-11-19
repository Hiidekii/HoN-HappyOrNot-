import numpy as np
from tensorflow.keras.models import load_model
import cv2

face_cascade = cv2.CascadeClassifier(
    'assets/haarcascade_frontalface_default.xml')

HEIGHT, WIDTH = 48, 48
dim = (HEIGHT, WIDTH)


initial_emotions = ['anger', 'disgust', 'fear',
                    'happiness', 'neutral', 'sadness', 'surprise']
initial_model = load_model("modelos/complet_faces_best_model20211110.h5")


def get_prediction(imagen):
    """Return predictions of each image provided by main app."""
    try:
        imagen = cv2.resize(imagen, dim)
        print(imagen.shape)
        # IMPRESCINDIBLE PARA MODELO GRAY
        imagen_g = imagen.reshape(1, HEIGHT, WIDTH, 1)
        print(imagen_g.shape)
        prediction = initial_model.predict(imagen_g)
        print(prediction)
        print(prediction.argmax())
        return initial_emotions[prediction.argmax()]

    except Exception as error:
        print(error)
