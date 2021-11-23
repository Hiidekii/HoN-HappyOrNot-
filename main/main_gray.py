from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from modules.hon import Ui_mainWindow
from modules.predict import get_prediction
from modules.video import Worker_Video
from modules.cam import Worker_Cam
from modules.sccap import Worker_Screen
import sys
import cv2

opencv_conf = {
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'fontScale': 1,
    'fontColor': (0, 255, 17),
    'lineType': 2
}

class HappyOrNot(QWidget, Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.images_tab.activateWindow()
        self.tabWidget.setCurrentWidget(
            self.tabWidget.findChild(QWidget, "acerca_de"))
        self.load_cam_button.clicked.connect(self.cam_view)
        self.load_video_button.clicked.connect(self.video_view)
        self.load_screen_button.clicked.connect(self.screen_view)
        self.counters = {}
        self.counters_vid = {}
        self.counters_screen = {}

    def cam_view(self):
        """Create a cam capture instance(Worker1) and begin the process"""
        self.counters = {
            'anger': {'val': 0, 'lcd': self.counter_anger},
            'disgust': {'val': 0, 'lcd': self.counter_disgust},
            'fear': {'val': 0, 'lcd': self.counter_fear},
            'happiness': {'val': 0, 'lcd': self.counter_happy},
            'neutral': {'val': 0, 'lcd': self.counter_neutral},
            'sadness': {'val': 0, 'lcd': self.counter_sad},
            'surprise': {'val': 0, 'lcd': self.counter_surp}
        }
        self.stop_cam_button.clicked.connect(self.cancelFeed)
        self.worker_cam = Worker_Cam(opencv_conf, welcome)
        self.worker_cam.start()
        self.worker_cam.ImageUpdate1.connect(self.imageUpdateSlot)

    def imageUpdateSlot(self, Image):
        self.cam_thread.setPixmap(QPixmap.fromImage(Image))

    def cancelFeed(self):
        self.worker_cam.stop()

    def video_view(self):
        """Create a video capture instance(Worker_Video) and begin the process"""
        self.counters_vid = {
            'anger': {'val': 0, 'lcd': self.counter_anger_vid},
            'disgust': {'val': 0, 'lcd': self.counter_disgust_vid},
            'fear': {'val': 0, 'lcd': self.counter_fear_vid},
            'happiness': {'val': 0, 'lcd': self.counter_happy_vid},
            'neutral': {'val': 0, 'lcd': self.counter_neutral_vid},
            'sadness': {'val': 0, 'lcd': self.counter_sad_vid},
            'surprise': {'val': 0, 'lcd': self.counter_surp_vid}
        }

        if self.video_path.toPlainText() != "":
            self.stop_video_button.clicked.connect(self.cancelFeed_video)
            self.worker_video = Worker_Video(opencv_conf, welcome)
            self.worker_video.start()
            self.worker_video.ImageUpdate.connect(self.imageUpdateSlot_video)
        else:
            self.video_path.append("Ningun Video")

    def imageUpdateSlot_video(self, Image):
        self.video_thread.setPixmap(QPixmap.fromImage(Image))

    def cancelFeed_video(self):
        self.worker_video.stop()

    def screen_view(self):
        """Create a screen capture instance(Worker_Screen) and begin the process"""
        self.counters_screen = {
            'anger': {'val': 0, 'lcd': self.counter_anger_screen},
            'disgust': {'val': 0, 'lcd': self.counter_disgust_screen},
            'fear': {'val': 0, 'lcd': self.counter_fear_screen},
            'happiness': {'val': 0, 'lcd': self.counter_happy_screen},
            'neutral': {'val': 0, 'lcd': self.counter_neutral_screen},
            'sadness': {'val': 0, 'lcd': self.counter_sad_screen},
            'surprise': {'val': 0, 'lcd': self.counter_surp_screen}
        }

        self.stop_screen_button.clicked.connect(self.cancelFeed_screen)
        self.worker_screen = Worker_Screen(opencv_conf, welcome)
        self.worker_screen.start()
        self.worker_screen.ImageUpdate2.connect(self.imageUpdateSlot_screen)

    def imageUpdateSlot_screen(self, Image):
        self.screen_thread.setPixmap(QPixmap.fromImage(Image))

    def cancelFeed_screen(self):
        self.worker_screen.stop()


if __name__ == "__main__":
    # Necesitamos siempre una aplicación
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('imgs/Hugging face.png'))
    welcome = HappyOrNot()
    welcome.show()

    # Inicia la aplicación
    app.exec()
