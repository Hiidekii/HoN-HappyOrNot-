from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage
from modules.predict import get_prediction
import cv2
import time
face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
# cv_conf = {
#             'font' : cv2.FONT_HERSHEY_SIMPLEX,
#             'fontScale' : 1,
#             'fontColor' : (0, 255, 17),
#             'lineType'  : 2
# }
class Worker1(QThread):
    ImageUpdate1 = pyqtSignal(QImage)
    def __init__(self, ocv_conf,welcome):
        super().__init__()
        self.config = ocv_conf
        self.welcome = welcome
        
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(-1)
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
                if i == 30:
                    if len(faces)>0:
                        self.faces_gray = []
                        for (x,y,w,h) in faces:
                            careto = {}
                            careto["gray"] = gray[y:y+h, x:x+w]
                            careto["pos"] = (x,y,w,h)
                            careto["pred"] = get_prediction(gray[y:y+h, x:x+w])
                            self.faces_gray.append(careto)
                            self.welcome.emotions_cam_reg.append(careto["pred"] + "  at  " + time.strftime("%X"))
                            print(careto["pred"])
                            print(self.welcome.counters[careto["pred"]]['val'])
                            self.welcome.counters[careto["pred"]]['val'] += 1
                            self.welcome.counters[careto["pred"]]['lcd'].display(self.welcome.counters[careto["pred"]]['val'])
                        
                        i= 0
                else:
                    i+=1
                for cara in self.faces_gray:
                    x,y,w,h = cara["pos"]
                    cv2.putText(Image,
                                cara["pred"], 
                                (x,y), 
                                self.config['font'], 
                                self.config['fontScale'],
                                self.config['fontColor'],
                                self.config['lineType'])

   
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate1.emit(Pic)
        Capture.release()
        cv2.destroyAllWindows()
        self.ImageUpdate1.disconnect()
        
    def stop(self):
        self.ThreadActive = False
        self.quit()