
# from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,QPushButton,QTextEdit
# from PyQt6.QtGui import QPixmap, QImage 
# from PyQt6.QtCore import QThread, pyqtSignal
# https://sensa.co/emoji/
# scrot

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from hon import Ui_mainWindow
import sys
import requests
import numpy as np
import cv2
from vidgear.gears import CamGear
from predict import get_prediction
import mss
import mss.tools
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
        self.load_video_button.clicked.connect(self.video_view)
        self.load_screen_button.clicked.connect(self.screen_view)
        self.counters = {}
        self.counters_vid={}
        self.counters_screen={}


    def cam_view(self):
        self.counters ={
            'anger': {'val':0, 'lcd':self.counter_anger},
            'disgust': {'val':0, 'lcd':self.counter_disgust}, 
            'fear': {'val':0, 'lcd':self.counter_fear}, 
            'happiness': {'val':0, 'lcd':self.counter_happy}, 
            'neutral': {'val':0, 'lcd':self.counter_neutral}, 
            'sadness': {'val':0, 'lcd':self.counter_sad}, 
            'surprise': {'val':0, 'lcd':self.counter_surp}
        }      
        self.stop_cam_button.clicked.connect(self.CancelFeed)

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

    def ImageUpdateSlot(self, Image):
        self.cam_thread.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()



    def video_view(self):
        self.counters_vid ={
            'anger': {'val':0, 'lcd':self.counter_anger_vid},
            'disgust': {'val':0, 'lcd':self.counter_disgust_vid}, 
            'fear': {'val':0, 'lcd':self.counter_fear_vid}, 
            'happiness': {'val':0, 'lcd':self.counter_happy_vid}, 
            'neutral': {'val':0, 'lcd':self.counter_neutral_vid}, 
            'sadness': {'val':0, 'lcd':self.counter_sad_vid}, 
            'surprise': {'val':0, 'lcd':self.counter_surp_vid}
        }

        if self.video_path.toPlainText() != "":      
            self.stop_video_button.clicked.connect(self.CancelFeed_video)

            self.Worker_Video = Worker_Video()
            self.Worker_Video.start()
            self.Worker_Video.ImageUpdate.connect(self.ImageUpdateSlot_video)
        else:
            self.video_path.append("Ningun Video")
    def ImageUpdateSlot_video(self, Image):
        self.video_thread.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed_video(self):
        self.Worker_Video.stop()

    
    def screen_view(self):
        self.counters_screen ={
            'anger': {'val':0, 'lcd':self.counter_anger_screen},
            'disgust': {'val':0, 'lcd':self.counter_disgust_screen}, 
            'fear': {'val':0, 'lcd':self.counter_fear_screen}, 
            'happiness': {'val':0, 'lcd':self.counter_happy_screen}, 
            'neutral': {'val':0, 'lcd':self.counter_neutral_screen}, 
            'sadness': {'val':0, 'lcd':self.counter_sad_screen}, 
            'surprise': {'val':0, 'lcd':self.counter_surp_screen}
        }

     
        self.stop_screen_button.clicked.connect(self.CancelFeed_video)

        self.Worker_Screen = Worker_Screen()
        self.Worker_Screen.start()
        self.Worker_Screen.ImageUpdate.connect(self.ImageUpdateSlot_screen)

    def ImageUpdateSlot_screen(self, Image):
        self.screen_thread.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed_video(self):
        self.Worker_Screen.stop()    



class Worker_Screen(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        with mss.mss() as sct:
            monitor_num = 1
            mon = sct.monitors[monitor_num]
            monitor = {
                    "top": mon["top"],
                    "left": mon["left"],
                    "width": mon["width"],
                    "height": mon["height"],
                    "mon": monitor_num,
            }
            #self.stream = CamGear(source=welcome.video_path.toPlainText(), stream_mode = True, logging=False).start()
            self.ThreadActive = True
            # Capture = cv2.VideoCapture(0)
            i = 0
            x,y,w,h = 0,0,0,0
            self.faces_gray = []
            while self.ThreadActive:
                img = sct.grab(monitor) # tomamos un pantallazo
                frame = np.array(img) 
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(255,0,0),2)                 
                # FlippedImage = cv2.flip(Image, 1)
                if i == 30:
                    if len(faces)>0:
                        self.faces_gray = []
                        for (x,y,w,h) in faces:
                            careto = {}
                            careto["gray"] = gray[y:y+h, x:x+w]
                            careto["pos"] = (x,y,w,h)
                            careto["pred"] = get_prediction(gray[y:y+h, x:x+w])
                            self.faces_gray.append(careto)
                            welcome.emotions_screen_reg.append(careto["pred"])
                            print(careto["pred"])
                            print(welcome.counters_screen[careto["pred"]]['val'])
                            welcome.counters_screen[careto["pred"]]['val'] += 1
                            welcome.counters_screen[careto["pred"]]['lcd'].display(welcome.counters_screen[careto["pred"]]['val'])
                        
                        i= 0
                else:
                    i+=1
                for cara in self.faces_gray:
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
            #self.stream.stop()
            cv2.destroyAllWindows()
            
    def stop(self):
        self.ThreadActive = False
        self.quit()



class Worker_Video(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.stream = CamGear(source=welcome.video_path.toPlainText(), stream_mode = True, logging=False).start()
        self.ThreadActive = True
        # Capture = cv2.VideoCapture(0)
        i = 0
        x,y,w,h = 0,0,0,0
        self.faces_gray = []
        while self.ThreadActive:
            frame = self.stream.read()
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(Image,(x,y),(x+w,y+h),(255,0,0),2)                 
            # FlippedImage = cv2.flip(Image, 1)
            if i == 30:
                if len(faces)>0:
                    self.faces_gray = []
                    for (x,y,w,h) in faces:
                        careto = {}
                        careto["gray"] = gray[y:y+h, x:x+w]
                        careto["pos"] = (x,y,w,h)
                        careto["pred"] = get_prediction(gray[y:y+h, x:x+w])
                        self.faces_gray.append(careto)
                        welcome.emotions_video_reg.append(careto["pred"])
                        print(careto["pred"])
                        print(welcome.counters_vid[careto["pred"]]['val'])
                        welcome.counters_vid[careto["pred"]]['val'] += 1
                        welcome.counters_vid[careto["pred"]]['lcd'].display(welcome.counters_vid[careto["pred"]]['val'])
                    
                    i= 0
            else:
                i+=1
            for cara in self.faces_gray:
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
        self.stream.stop()
        cv2.destroyAllWindows()
        
    def stop(self):
        self.ThreadActive = False
        self.quit()



class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        i = 0
        x,y,w,h = 0,0,0,0
        self.faces_gray = []
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(255,0,0),2)                 
                # FlippedImage = cv2.flip(Image, 1)
                if i == 30:
                    if len(faces)>0:
                        self.faces_gray = []
                        for (x,y,w,h) in faces:
                            careto = {}
                            careto["gray"] = gray[y:y+h, x:x+w]
                            careto["pos"] = (x,y,w,h)
                            careto["pred"] = get_prediction(gray[y:y+h, x:x+w])
                            self.faces_gray.append(careto)
                            welcome.emotions_cam_reg.append(careto["pred"])
                            print(careto["pred"])
                            print(welcome.counters[careto["pred"]]['val'])
                            welcome.counters[careto["pred"]]['val'] += 1
                            welcome.counters[careto["pred"]]['lcd'].display(welcome.counters[careto["pred"]]['val'])
                        
                        i= 0
                else:
                    i+=1
                for cara in self.faces_gray:
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