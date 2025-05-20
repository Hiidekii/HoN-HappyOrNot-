from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage
from modules.predict import get_prediction
import cv2
import time
import mss
import numpy as np
import os

cascade_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

class Worker_Screen(QThread):
    ImageUpdate2 = pyqtSignal(QImage)

    def __init__(self, ocv_conf, welcome):
        super().__init__()
        self.config = ocv_conf
        self.welcome = welcome

    def run(self):
        with mss.mss() as sct:
            monitor_num = 1  # Tu pantalla de laptop
            mon = sct.monitors[monitor_num]

            # ✅ Forzamos resolución segura de captura (puedes ajustarla)
            monitor = {
                "top": 0,
                "left": 0,
                "width": 1280,
                "height": 720,
                "mon": monitor_num,
            }

            self.ThreadActive = True
            i = 0
            self.faces_gray = []

            while self.ThreadActive:
                img = sct.grab(monitor)

                # ✅ Eliminamos canal alpha (BGRA → RGB)
                frame = np.array(img, dtype=np.uint8)[:, :, :3]
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                Image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(Image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if i == 30:
                    if len(faces) > 0:
                        self.faces_gray = []
                        for (x, y, w, h) in faces:
                            careto = {}
                            roi_gray = gray[y:y + h, x:x + w]
                            face_resized = cv2.resize(roi_gray, (64, 64))
                            careto["gray"] = face_resized
                            careto["pos"] = (x, y, w, h)
                            careto["pred"] = get_prediction(face_resized) or "unknown"
                            self.faces_gray.append(careto)
                            self.welcome.emotions_screen_reg.append(
                                f"{careto['pred']}  at  {time.strftime('%X')}"
                            )
                            self.welcome.counters_screen[careto["pred"]]['val'] += 1
                            self.welcome.counters_screen[careto["pred"]]['lcd'].display(
                                self.welcome.counters_screen[careto["pred"]]['val']
                            )
                        i = 0
                else:
                    i += 1

                for cara in self.faces_gray:
                    x, y, w, h = cara["pos"]
                    cv2.putText(
                        Image,
                        str(cara["pred"]),
                        (x, y),
                        self.config['font'],
                        self.config['fontScale'],
                        self.config['fontColor'],
                        self.config['lineType']
                    )

                ConvertToQtFormat = QImage(
                    Image.data, Image.shape[1], Image.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(
                    640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate2.emit(Pic)

            cv2.destroyAllWindows()

    def stop(self):
        self.ThreadActive = False
        self.quit()
