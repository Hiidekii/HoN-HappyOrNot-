import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

cascade_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

HEIGHT, WIDTH = 48,48
dim=(HEIGHT, WIDTH)



# Estas variables son para pasar la image por varios modelos.
# other_emotions =  ['anger', 'disgust', 'fear', 'neutral']
# initial_emotions = ['happiness', 'other', 'sadness', 'surprise']
# initial_model = load_model("modelos/initial_emotions20211116_v4[sad-hap-sur].h5")
# others_model = load_model("modelos/other_emotions20211116_v4[fea-ang-dis-neu].h5")

# 4 emociones
# initial_emotions = ['happiness', 'neutral', 'sadness', 'surprise']
# initial_model =load_model("modelos/no_dig_no_fear20211117_v1.h5")

# 7 emociones
initial_emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
initial_model =load_model("modelos/complet_faces_best_model20211110.h5")


# def get_face(img):
#     try:
#         print(face_cascade.detectMultiScale(img, 1.3, 5))
#         x, y, w, h = face_cascade.detectMultiScale(img, 1.3, 5)
#         return img[y:y+h, x:x+w]
#     except:
#         print("no estoy decetando rostro")

# i=0

def get_prediction(imagen):
    try:
        # imagen = get_face(img)
        #imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = cv2.resize(imagen, dim)
        print(imagen.shape)
        imagen_g =imagen.reshape(1,HEIGHT, WIDTH,1) #IMPRESCINDIBLE PARA MODELO GRAY
        #imagen_g =imagen.reshape(1,HEIGHT, WIDTH,1)
        print(imagen_g.shape)
        #cv2.imwrite('pruebas/waka.jpg', imagen)
        prediction = initial_model.predict(imagen_g)
        print(prediction)
        print(prediction.argmax())
        return initial_emotions[prediction.argmax()]

        #Este codigo lo use para probar pasar la iagen por varios modelos.
        # if initial_emotions[prediction.argmax()] != 'other':
        #     return initial_emotions[prediction.argmax()]
        # else:
        #     prediction = others_model.predict(imagen_g)
        #     return other_emotions[prediction.argmax()]

    except Exception as error:
        print(error)
