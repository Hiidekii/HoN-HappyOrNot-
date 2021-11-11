
# from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,QPushButton,QTextEdit
# from PyQt6.QtGui import QPixmap, QImage 
# from PyQt6.QtCore import QThread, pyqtSignal

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from hon import Ui_mainWindow
import sys
import requests
import numpy as np
import cv2
from predict import get_prediction
# from html2text import html2text

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (0, 255, 17)
lineType               = 2
face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')

class HappyOrNot(QWidget,Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.images_tab.activateWindow()
        self.tabWidget.setCurrentWidget(self.tabWidget.findChild(QWidget,"acerca_de"))
        self.load_cam_button.clicked.connect(self.cam_view)




    def cam_view(self):
       
        # self.VBL = QVBoxLayout()

        # self.VBL.addWidget(self.cam_thread)

        
        self.stop_cam_button.clicked.connect(self.CancelFeed)
        # self.VBL.addWidget(self.stop_cam_button)

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        
        # self.VBL.addWidget(self.emotions_cam_reg)
        
        # self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.cam_thread.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()



class Worker1(QThread):
    # font                   = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale              = 0.75
    # fontColor              = (0, 255, 17)
    # lineType               = 2
    # face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')

    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        i = 0
        texto = ""
        x,y,w,h = 0,0,0,0
        faces_gray = []
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(255,0,0),2)                 
                
                FlippedImage = cv2.flip(Image, 1)

                if i == 30:
                    if len(faces)>0:
                        faces_gray = []
                        for (x,y,w,h) in faces:
                            careto = {}
                            careto["gray"] = gray[y:y+h, x:x+w]
                            careto["pos"] = (x,y,w,h)
                            careto["pred"] = get_prediction(gray[y:y+h, x:x+w])
                            faces_gray.append(careto)
                        i= 0
                else:
                    i+=1
                for cara in faces_gray:
                    x,y,w,h = cara["pos"]
                    cv2.putText(Image,
                                cara["pred"], 
                                (x,y), 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)

   
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
        Capture.release()
        cv2.destroyAllWindows()
        
    def stop(self):
        self.ThreadActive = False
        self.quit()










if __name__ == "__main__":
    # Necesitamos siempre una aplicación
    app = QApplication(sys.argv)

    # Pueden haber diferentes widgets
    welcome = HappyOrNot()
    welcome.show()

    # Inicia la aplicación
    app.exec()