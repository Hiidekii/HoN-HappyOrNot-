import cv2
from os import listdir
import numpy as np


face_cascade = cv2.CascadeClassifier(
    '../main/assets/haarcascade_frontalface_default.xml')
all_emotions = ['sadness', 'happiness', 'surprise',
                'neutral', 'fear', 'disgust', 'anger']
paths_default = ["data/all/train/", "data/all/test/"]
ignore = ["morralla", ".DS_Store", "contempt"]

class Data_cut():
    def __init__(self, paths=paths_default,emotions = all_emotions,h=48,w=48):
        self.dim = (h,w)
        self.imgs = []
        self.state = []
        self.paths = paths
        if len(emotions) == 7:
            self.emotions = emotions
        elif len(emotions) < 7:
            self.emotions = [x for x in all_emotions if x in emotions]
            self.emotions.append('other')
        else: 
            print("emotions list too large, max 7")
        self.other_emotions = [
            x for x in all_emotions if x not in self.emotions]

    def get_emotion(self,x,initial):
        if initial:
            if x in self.emotions:
                return x
            else:
                return "other"
        else: 
            if x in self.other_emotions:
                return x


    def load_imgs(self):
        for path in self.paths:
            for item in listdir(path):
                if item not in ignore:
                    self.imgs.extend([{"path": f"{path}{item}/{p}", "emotion": self.get_emotion(item, True)}
                                for p in listdir(f"{path}{item}")])


    def recortar(self):
        imgs_ = []
        state_ = []
        for p in self.imgs:
            temp = cv2.imread(p["path"],0)
            faces = face_cascade.detectMultiScale(temp, 1.1, 5) #1.1 factor scale, cuanto  disminuye imagen entre pasos
                                                                #5 MIN nEIGHBORS cuantos vecinos cada posible 
                                                                # rectangulo considerar para prediccion
                                                                # (valor minimo px de la cara detectada)
                                                                # (valor maximo px)
                    
            if len(faces)==1:
                for (x,y,w,h) in faces:
                    recortada = temp[y:y+h, x:x+w]
                    recortada = cv2.resize(recortada,self.dim)
                    imgs_.append(recortada)
                    state_.append(p["emotion"])
            
                
        imgs_ = [el/255 for el in imgs_]
        imgs_= np.array(imgs_)
        self.imgs = imgs_
        self.state = state_ 


