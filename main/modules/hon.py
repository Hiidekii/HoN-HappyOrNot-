# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(901, 671)
        mainWindow.setMinimumSize(QtCore.QSize(901, 671))
        mainWindow.setMaximumSize(QtCore.QSize(901, 671))
        mainWindow.setAutoFillBackground(False)
        mainWindow.setStyleSheet("#mainWindow{\n"
"    background-color : #FC6941\n"
"}\n"
"\n"
".QPushButton{\n"
"    background-color: #FC6941\n"
"}\n"
".TtabWidget{\n"
"background-color: #FFFFFF\n"
"}\n"
".QFrame{\n"
"background-color: #FC6941\n"
"}\n"
"\n"
".QLCDNumber{\n"
"    background-color: #FFFFFF\n"
"}")
        self.tabWidget = QtWidgets.QTabWidget(parent=mainWindow)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 881, 651))
        self.tabWidget.setMinimumSize(QtCore.QSize(881, 651))
        self.tabWidget.setMaximumSize(QtCore.QSize(881, 651))
        self.tabWidget.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        self.tabWidget.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("#tabWidget{\n"
"background-color: #FFFFFF\n"
"}")
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        self.tabWidget.setIconSize(QtCore.QSize(20, 20))
        self.tabWidget.setElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.tabWidget.setObjectName("tabWidget")
        self.images_tab = QtWidgets.QWidget()
        self.images_tab.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        self.images_tab.setToolTip("")
        self.images_tab.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.images_tab.setObjectName("images_tab")
        self.image_thread = QtWidgets.QLabel(parent=self.images_tab)
        self.image_thread.setGeometry(QtCore.QRect(10, 10, 400, 400))
        self.image_thread.setMinimumSize(QtCore.QSize(400, 400))
        self.image_thread.setMaximumSize(QtCore.QSize(400, 400))
        self.image_thread.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.image_thread.setText("")
        self.image_thread.setObjectName("image_thread")
        self.toolButton_image = QtWidgets.QToolButton(parent=self.images_tab)
        self.toolButton_image.setGeometry(QtCore.QRect(625, 10, 31, 31))
        self.toolButton_image.setObjectName("toolButton_image")
        self.emotions_image_reg = QtWidgets.QTextEdit(parent=self.images_tab)
        self.emotions_image_reg.setGeometry(QtCore.QRect(430, 120, 211, 290))
        self.emotions_image_reg.setMinimumSize(QtCore.QSize(211, 290))
        self.emotions_image_reg.setMaximumSize(QtCore.QSize(211, 290))
        self.emotions_image_reg.setObjectName("emotions_image_reg")
        self.load_image_button = QtWidgets.QPushButton(parent=self.images_tab)
        self.load_image_button.setGeometry(QtCore.QRect(480, 50, 101, 41))
        self.load_image_button.setObjectName("load_image_button")
        self.image_path = QtWidgets.QTextEdit(parent=self.images_tab)
        self.image_path.setGeometry(QtCore.QRect(420, 10, 201, 31))
        self.image_path.setObjectName("image_path")
        self.labelimage = QtWidgets.QLabel(parent=self.images_tab)
        self.labelimage.setGeometry(QtCore.QRect(430, 90, 151, 21))
        self.labelimage.setObjectName("labelimage")
        self.tabWidget.addTab(self.images_tab, "")
        self.capture_tab = QtWidgets.QWidget()
        self.capture_tab.setObjectName("capture_tab")
        self.emotions_screen_reg = QtWidgets.QTextEdit(parent=self.capture_tab)
        self.emotions_screen_reg.setGeometry(QtCore.QRect(660, 80, 211, 311))
        self.emotions_screen_reg.setMinimumSize(QtCore.QSize(211, 311))
        self.emotions_screen_reg.setMaximumSize(QtCore.QSize(211, 311))
        self.emotions_screen_reg.setObjectName("emotions_screen_reg")
        self.screen_thread = QtWidgets.QLabel(parent=self.capture_tab)
        self.screen_thread.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.screen_thread.setMinimumSize(QtCore.QSize(640, 480))
        self.screen_thread.setMaximumSize(QtCore.QSize(640, 480))
        self.screen_thread.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.screen_thread.setText("")
        self.screen_thread.setObjectName("screen_thread")
        self.stop_screen_button = QtWidgets.QPushButton(parent=self.capture_tab)
        self.stop_screen_button.setGeometry(QtCore.QRect(710, 400, 111, 41))
        self.stop_screen_button.setObjectName("stop_screen_button")
        self.load_screen_button = QtWidgets.QPushButton(parent=self.capture_tab)
        self.load_screen_button.setGeometry(QtCore.QRect(710, 10, 121, 41))
        self.load_screen_button.setObjectName("load_screen_button")
        self.contadores_screen = QtWidgets.QFrame(parent=self.capture_tab)
        self.contadores_screen.setGeometry(QtCore.QRect(50, 500, 781, 111))
        self.contadores_screen.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.contadores_screen.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.contadores_screen.setObjectName("contadores_screen")
        self.label_screen_happy = QtWidgets.QLabel(parent=self.contadores_screen)
        self.label_screen_happy.setGeometry(QtCore.QRect(50, 30, 50, 50))
        self.label_screen_happy.setText("")
        self.label_screen_happy.setPixmap(QtGui.QPixmap(":/imgs/imgs/Smiling face with smiling eyes.png"))
        self.label_screen_happy.setScaledContents(True)
        self.label_screen_happy.setWordWrap(False)
        self.label_screen_happy.setObjectName("label_screen_happy")
        self.label_screen_surprise = QtWidgets.QLabel(parent=self.contadores_screen)
        self.label_screen_surprise.setGeometry(QtCore.QRect(250, 30, 50, 50))
        self.label_screen_surprise.setText("")
        self.label_screen_surprise.setPixmap(QtGui.QPixmap(":/imgs/imgs/Astonished face.png"))
        self.label_screen_surprise.setScaledContents(True)
        self.label_screen_surprise.setObjectName("label_screen_surprise")
        self.label_screen_sad = QtWidgets.QLabel(parent=self.contadores_screen)
        self.label_screen_sad.setGeometry(QtCore.QRect(150, 30, 50, 50))
        self.label_screen_sad.setText("")
        self.label_screen_sad.setPixmap(QtGui.QPixmap(":/imgs/imgs/Slightly frowning face.png"))
        self.label_screen_sad.setScaledContents(True)
        self.label_screen_sad.setObjectName("label_screen_sad")
        self.label_screen_anger = QtWidgets.QLabel(parent=self.contadores_screen)
        self.label_screen_anger.setGeometry(QtCore.QRect(680, 30, 50, 50))
        self.label_screen_anger.setText("")
        self.label_screen_anger.setPixmap(QtGui.QPixmap(":/imgs/imgs/Angry face.png"))
        self.label_screen_anger.setScaledContents(True)
        self.label_screen_anger.setObjectName("label_screen_anger")
        self.counter_fear_screen = QtWidgets.QLCDNumber(parent=self.contadores_screen)
        self.counter_fear_screen.setGeometry(QtCore.QRect(350, 80, 64, 23))
        self.counter_fear_screen.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_fear_screen.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_fear_screen.setLineWidth(1)
        self.counter_fear_screen.setMidLineWidth(0)
        self.counter_fear_screen.setSmallDecimalPoint(False)
        self.counter_fear_screen.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_fear_screen.setObjectName("counter_fear_screen")
        self.counter_neutral_screen = QtWidgets.QLCDNumber(parent=self.contadores_screen)
        self.counter_neutral_screen.setGeometry(QtCore.QRect(450, 80, 64, 23))
        self.counter_neutral_screen.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_neutral_screen.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_neutral_screen.setLineWidth(1)
        self.counter_neutral_screen.setMidLineWidth(0)
        self.counter_neutral_screen.setSmallDecimalPoint(False)
        self.counter_neutral_screen.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_neutral_screen.setObjectName("counter_neutral_screen")
        self.label_screen_neutral = QtWidgets.QLabel(parent=self.contadores_screen)
        self.label_screen_neutral.setGeometry(QtCore.QRect(460, 30, 50, 50))
        self.label_screen_neutral.setText("")
        self.label_screen_neutral.setPixmap(QtGui.QPixmap(":/imgs/imgs/Neutral face.png"))
        self.label_screen_neutral.setScaledContents(True)
        self.label_screen_neutral.setObjectName("label_screen_neutral")
        self.counter_surp_screen = QtWidgets.QLCDNumber(parent=self.contadores_screen)
        self.counter_surp_screen.setGeometry(QtCore.QRect(240, 80, 64, 23))
        self.counter_surp_screen.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_surp_screen.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_surp_screen.setLineWidth(1)
        self.counter_surp_screen.setMidLineWidth(0)
        self.counter_surp_screen.setSmallDecimalPoint(False)
        self.counter_surp_screen.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_surp_screen.setObjectName("counter_surp_screen")
        self.counter_happy_screen = QtWidgets.QLCDNumber(parent=self.contadores_screen)
        self.counter_happy_screen.setGeometry(QtCore.QRect(40, 80, 64, 23))
        self.counter_happy_screen.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_happy_screen.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_happy_screen.setLineWidth(1)
        self.counter_happy_screen.setMidLineWidth(0)
        self.counter_happy_screen.setSmallDecimalPoint(False)
        self.counter_happy_screen.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_happy_screen.setObjectName("counter_happy_screen")
        self.counter_sad_screen = QtWidgets.QLCDNumber(parent=self.contadores_screen)
        self.counter_sad_screen.setGeometry(QtCore.QRect(140, 80, 64, 23))
        self.counter_sad_screen.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_sad_screen.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_sad_screen.setLineWidth(1)
        self.counter_sad_screen.setMidLineWidth(0)
        self.counter_sad_screen.setSmallDecimalPoint(False)
        self.counter_sad_screen.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_sad_screen.setObjectName("counter_sad_screen")
        self.counter_disgust_screen = QtWidgets.QLCDNumber(parent=self.contadores_screen)
        self.counter_disgust_screen.setGeometry(QtCore.QRect(560, 80, 64, 23))
        self.counter_disgust_screen.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_disgust_screen.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_disgust_screen.setLineWidth(1)
        self.counter_disgust_screen.setMidLineWidth(0)
        self.counter_disgust_screen.setSmallDecimalPoint(False)
        self.counter_disgust_screen.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_disgust_screen.setObjectName("counter_disgust_screen")
        self.counter_anger_screen = QtWidgets.QLCDNumber(parent=self.contadores_screen)
        self.counter_anger_screen.setGeometry(QtCore.QRect(670, 80, 64, 23))
        self.counter_anger_screen.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_anger_screen.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_anger_screen.setLineWidth(1)
        self.counter_anger_screen.setMidLineWidth(0)
        self.counter_anger_screen.setSmallDecimalPoint(False)
        self.counter_anger_screen.setMode(QtWidgets.QLCDNumber.Mode.Dec)
        self.counter_anger_screen.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_anger_screen.setObjectName("counter_anger_screen")
        self.label_screen_fear = QtWidgets.QLabel(parent=self.contadores_screen)
        self.label_screen_fear.setGeometry(QtCore.QRect(360, 30, 50, 50))
        self.label_screen_fear.setText("")
        self.label_screen_fear.setPixmap(QtGui.QPixmap(":/imgs/imgs/Fearful face.png"))
        self.label_screen_fear.setScaledContents(True)
        self.label_screen_fear.setObjectName("label_screen_fear")
        self.label_screen_disgust = QtWidgets.QLabel(parent=self.contadores_screen)
        self.label_screen_disgust.setGeometry(QtCore.QRect(570, 30, 50, 50))
        self.label_screen_disgust.setText("")
        self.label_screen_disgust.setPixmap(QtGui.QPixmap(":/imgs/imgs/Pensive face.png"))
        self.label_screen_disgust.setScaledContents(True)
        self.label_screen_disgust.setObjectName("label_screen_disgust")
        self.label_screen_emometer = QtWidgets.QLabel(parent=self.contadores_screen)
        self.label_screen_emometer.setGeometry(QtCore.QRect(200, -4, 391, 41))
        self.label_screen_emometer.setScaledContents(True)
        self.label_screen_emometer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_screen_emometer.setWordWrap(False)
        self.label_screen_emometer.setObjectName("label_screen_emometer")
        self.labelscreen1 = QtWidgets.QLabel(parent=self.capture_tab)
        self.labelscreen1.setGeometry(QtCore.QRect(700, 60, 151, 21))
        self.labelscreen1.setObjectName("labelscreen1")
        self.tabWidget.addTab(self.capture_tab, "")
        self.video_tab = QtWidgets.QWidget()
        self.video_tab.setObjectName("video_tab")
        self.video_thread = QtWidgets.QLabel(parent=self.video_tab)
        self.video_thread.setGeometry(QtCore.QRect(10, 50, 640, 440))
        self.video_thread.setMinimumSize(QtCore.QSize(640, 440))
        self.video_thread.setMaximumSize(QtCore.QSize(640, 440))
        self.video_thread.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.video_thread.setText("")
        self.video_thread.setObjectName("video_thread")
        self.load_video_button = QtWidgets.QPushButton(parent=self.video_tab)
        self.load_video_button.setGeometry(QtCore.QRect(710, 50, 101, 41))
        self.load_video_button.setObjectName("load_video_button")
        self.video_path = QtWidgets.QTextEdit(parent=self.video_tab)
        self.video_path.setGeometry(QtCore.QRect(10, 10, 821, 31))
        self.video_path.setObjectName("video_path")
        self.emotions_video_reg = QtWidgets.QTextEdit(parent=self.video_tab)
        self.emotions_video_reg.setGeometry(QtCore.QRect(660, 120, 211, 311))
        self.emotions_video_reg.setMinimumSize(QtCore.QSize(211, 311))
        self.emotions_video_reg.setMaximumSize(QtCore.QSize(211, 311))
        self.emotions_video_reg.setObjectName("emotions_video_reg")
        self.labelcaamvideo = QtWidgets.QLabel(parent=self.video_tab)
        self.labelcaamvideo.setGeometry(QtCore.QRect(690, 100, 151, 21))
        self.labelcaamvideo.setObjectName("labelcaamvideo")
        self.toolButton = QtWidgets.QToolButton(parent=self.video_tab)
        self.toolButton.setGeometry(QtCore.QRect(840, 10, 31, 31))
        self.toolButton.setObjectName("toolButton")
        self.stop_video_button = QtWidgets.QPushButton(parent=self.video_tab)
        self.stop_video_button.setGeometry(QtCore.QRect(710, 440, 111, 41))
        self.stop_video_button.setObjectName("stop_video_button")
        self.contadores_vid = QtWidgets.QFrame(parent=self.video_tab)
        self.contadores_vid.setGeometry(QtCore.QRect(50, 500, 781, 111))
        self.contadores_vid.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.contadores_vid.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.contadores_vid.setObjectName("contadores_vid")
        self.label_video_happy = QtWidgets.QLabel(parent=self.contadores_vid)
        self.label_video_happy.setGeometry(QtCore.QRect(50, 30, 50, 50))
        self.label_video_happy.setText("")
        self.label_video_happy.setPixmap(QtGui.QPixmap(":/imgs/imgs/Smiling face with smiling eyes.png"))
        self.label_video_happy.setScaledContents(True)
        self.label_video_happy.setWordWrap(False)
        self.label_video_happy.setObjectName("label_video_happy")
        self.label_video_surprise = QtWidgets.QLabel(parent=self.contadores_vid)
        self.label_video_surprise.setGeometry(QtCore.QRect(250, 30, 50, 50))
        self.label_video_surprise.setText("")
        self.label_video_surprise.setPixmap(QtGui.QPixmap(":/imgs/imgs/Astonished face.png"))
        self.label_video_surprise.setScaledContents(True)
        self.label_video_surprise.setObjectName("label_video_surprise")
        self.label_video_sad = QtWidgets.QLabel(parent=self.contadores_vid)
        self.label_video_sad.setGeometry(QtCore.QRect(150, 30, 50, 50))
        self.label_video_sad.setText("")
        self.label_video_sad.setPixmap(QtGui.QPixmap(":/imgs/imgs/Frowning face.png"))
        self.label_video_sad.setScaledContents(True)
        self.label_video_sad.setObjectName("label_video_sad")
        self.label_video_anger = QtWidgets.QLabel(parent=self.contadores_vid)
        self.label_video_anger.setGeometry(QtCore.QRect(680, 30, 50, 50))
        self.label_video_anger.setText("")
        self.label_video_anger.setPixmap(QtGui.QPixmap(":/imgs/imgs/Angry face.png"))
        self.label_video_anger.setScaledContents(True)
        self.label_video_anger.setObjectName("label_video_anger")
        self.counter_fear_vid = QtWidgets.QLCDNumber(parent=self.contadores_vid)
        self.counter_fear_vid.setGeometry(QtCore.QRect(350, 80, 64, 23))
        self.counter_fear_vid.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_fear_vid.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_fear_vid.setLineWidth(1)
        self.counter_fear_vid.setMidLineWidth(0)
        self.counter_fear_vid.setSmallDecimalPoint(False)
        self.counter_fear_vid.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_fear_vid.setObjectName("counter_fear_vid")
        self.counter_neutral_vid = QtWidgets.QLCDNumber(parent=self.contadores_vid)
        self.counter_neutral_vid.setGeometry(QtCore.QRect(450, 80, 64, 23))
        self.counter_neutral_vid.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_neutral_vid.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_neutral_vid.setLineWidth(1)
        self.counter_neutral_vid.setMidLineWidth(0)
        self.counter_neutral_vid.setSmallDecimalPoint(False)
        self.counter_neutral_vid.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_neutral_vid.setObjectName("counter_neutral_vid")
        self.label_video_neutral = QtWidgets.QLabel(parent=self.contadores_vid)
        self.label_video_neutral.setGeometry(QtCore.QRect(460, 30, 50, 50))
        self.label_video_neutral.setText("")
        self.label_video_neutral.setPixmap(QtGui.QPixmap(":/imgs/imgs/Neutral face.png"))
        self.label_video_neutral.setScaledContents(True)
        self.label_video_neutral.setObjectName("label_video_neutral")
        self.counter_surp_vid = QtWidgets.QLCDNumber(parent=self.contadores_vid)
        self.counter_surp_vid.setGeometry(QtCore.QRect(240, 80, 64, 23))
        self.counter_surp_vid.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_surp_vid.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_surp_vid.setLineWidth(1)
        self.counter_surp_vid.setMidLineWidth(0)
        self.counter_surp_vid.setSmallDecimalPoint(False)
        self.counter_surp_vid.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_surp_vid.setObjectName("counter_surp_vid")
        self.counter_happy_vid = QtWidgets.QLCDNumber(parent=self.contadores_vid)
        self.counter_happy_vid.setGeometry(QtCore.QRect(40, 80, 64, 23))
        self.counter_happy_vid.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_happy_vid.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_happy_vid.setLineWidth(1)
        self.counter_happy_vid.setMidLineWidth(0)
        self.counter_happy_vid.setSmallDecimalPoint(False)
        self.counter_happy_vid.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_happy_vid.setObjectName("counter_happy_vid")
        self.counter_sad_vid = QtWidgets.QLCDNumber(parent=self.contadores_vid)
        self.counter_sad_vid.setGeometry(QtCore.QRect(140, 80, 64, 23))
        self.counter_sad_vid.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_sad_vid.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_sad_vid.setLineWidth(1)
        self.counter_sad_vid.setMidLineWidth(0)
        self.counter_sad_vid.setSmallDecimalPoint(False)
        self.counter_sad_vid.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_sad_vid.setObjectName("counter_sad_vid")
        self.counter_disgust_vid = QtWidgets.QLCDNumber(parent=self.contadores_vid)
        self.counter_disgust_vid.setGeometry(QtCore.QRect(560, 80, 64, 23))
        self.counter_disgust_vid.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_disgust_vid.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_disgust_vid.setLineWidth(1)
        self.counter_disgust_vid.setMidLineWidth(0)
        self.counter_disgust_vid.setSmallDecimalPoint(False)
        self.counter_disgust_vid.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_disgust_vid.setObjectName("counter_disgust_vid")
        self.counter_anger_vid = QtWidgets.QLCDNumber(parent=self.contadores_vid)
        self.counter_anger_vid.setGeometry(QtCore.QRect(670, 80, 64, 23))
        self.counter_anger_vid.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_anger_vid.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_anger_vid.setLineWidth(1)
        self.counter_anger_vid.setMidLineWidth(0)
        self.counter_anger_vid.setSmallDecimalPoint(False)
        self.counter_anger_vid.setMode(QtWidgets.QLCDNumber.Mode.Dec)
        self.counter_anger_vid.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_anger_vid.setObjectName("counter_anger_vid")
        self.label_video_fear = QtWidgets.QLabel(parent=self.contadores_vid)
        self.label_video_fear.setGeometry(QtCore.QRect(360, 30, 50, 50))
        self.label_video_fear.setText("")
        self.label_video_fear.setPixmap(QtGui.QPixmap(":/imgs/imgs/Fearful face.png"))
        self.label_video_fear.setScaledContents(True)
        self.label_video_fear.setObjectName("label_video_fear")
        self.label_video_disgust = QtWidgets.QLabel(parent=self.contadores_vid)
        self.label_video_disgust.setGeometry(QtCore.QRect(570, 30, 50, 50))
        self.label_video_disgust.setText("")
        self.label_video_disgust.setPixmap(QtGui.QPixmap(":/imgs/imgs/Pensive face.png"))
        self.label_video_disgust.setScaledContents(True)
        self.label_video_disgust.setObjectName("label_video_disgust")
        self.label_video_emometer = QtWidgets.QLabel(parent=self.contadores_vid)
        self.label_video_emometer.setGeometry(QtCore.QRect(200, -4, 391, 41))
        self.label_video_emometer.setScaledContents(True)
        self.label_video_emometer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_video_emometer.setWordWrap(False)
        self.label_video_emometer.setObjectName("label_video_emometer")
        self.tabWidget.addTab(self.video_tab, "")
        self.cam_tab = QtWidgets.QWidget()
        self.cam_tab.setObjectName("cam_tab")
        self.cam_thread = QtWidgets.QLabel(parent=self.cam_tab)
        self.cam_thread.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.cam_thread.setMinimumSize(QtCore.QSize(640, 480))
        self.cam_thread.setMaximumSize(QtCore.QSize(640, 480))
        self.cam_thread.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.cam_thread.setText("")
        self.cam_thread.setObjectName("cam_thread")
        self.load_cam_button = QtWidgets.QPushButton(parent=self.cam_tab)
        self.load_cam_button.setGeometry(QtCore.QRect(710, 10, 121, 41))
        self.load_cam_button.setObjectName("load_cam_button")
        self.labelcaam1 = QtWidgets.QLabel(parent=self.cam_tab)
        self.labelcaam1.setGeometry(QtCore.QRect(700, 60, 151, 21))
        self.labelcaam1.setObjectName("labelcaam1")
        self.emotions_cam_reg = QtWidgets.QTextEdit(parent=self.cam_tab)
        self.emotions_cam_reg.setGeometry(QtCore.QRect(660, 80, 211, 311))
        self.emotions_cam_reg.setMinimumSize(QtCore.QSize(211, 311))
        self.emotions_cam_reg.setMaximumSize(QtCore.QSize(211, 311))
        self.emotions_cam_reg.setObjectName("emotions_cam_reg")
        self.stop_cam_button = QtWidgets.QPushButton(parent=self.cam_tab)
        self.stop_cam_button.setGeometry(QtCore.QRect(710, 400, 111, 41))
        self.stop_cam_button.setObjectName("stop_cam_button")
        self.contadores = QtWidgets.QFrame(parent=self.cam_tab)
        self.contadores.setGeometry(QtCore.QRect(50, 500, 781, 111))
        self.contadores.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.contadores.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.contadores.setObjectName("contadores")
        self.label_cam_happy = QtWidgets.QLabel(parent=self.contadores)
        self.label_cam_happy.setGeometry(QtCore.QRect(50, 30, 50, 50))
        self.label_cam_happy.setText("")
        self.label_cam_happy.setPixmap(QtGui.QPixmap(":/imgs/imgs/Smiling face with smiling eyes.png"))
        self.label_cam_happy.setScaledContents(True)
        self.label_cam_happy.setWordWrap(False)
        self.label_cam_happy.setObjectName("label_cam_happy")
        self.label_cam_surprise = QtWidgets.QLabel(parent=self.contadores)
        self.label_cam_surprise.setGeometry(QtCore.QRect(250, 30, 50, 50))
        self.label_cam_surprise.setText("")
        self.label_cam_surprise.setPixmap(QtGui.QPixmap(":/imgs/imgs/Astonished face.png"))
        self.label_cam_surprise.setScaledContents(True)
        self.label_cam_surprise.setObjectName("label_cam_surprise")
        self.label_cam_sad = QtWidgets.QLabel(parent=self.contadores)
        self.label_cam_sad.setGeometry(QtCore.QRect(150, 30, 50, 50))
        self.label_cam_sad.setText("")
        self.label_cam_sad.setPixmap(QtGui.QPixmap(":/imgs/imgs/Frowning face.png"))
        self.label_cam_sad.setScaledContents(True)
        self.label_cam_sad.setObjectName("label_cam_sad")
        self.label_cam_anger = QtWidgets.QLabel(parent=self.contadores)
        self.label_cam_anger.setGeometry(QtCore.QRect(680, 30, 50, 50))
        self.label_cam_anger.setText("")
        self.label_cam_anger.setPixmap(QtGui.QPixmap(":/imgs/imgs/Angry face.png"))
        self.label_cam_anger.setScaledContents(True)
        self.label_cam_anger.setObjectName("label_cam_anger")
        self.counter_fear = QtWidgets.QLCDNumber(parent=self.contadores)
        self.counter_fear.setGeometry(QtCore.QRect(350, 80, 64, 23))
        self.counter_fear.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_fear.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_fear.setLineWidth(1)
        self.counter_fear.setMidLineWidth(0)
        self.counter_fear.setSmallDecimalPoint(False)
        self.counter_fear.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_fear.setObjectName("counter_fear")
        self.counter_neutral = QtWidgets.QLCDNumber(parent=self.contadores)
        self.counter_neutral.setGeometry(QtCore.QRect(450, 80, 64, 23))
        self.counter_neutral.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_neutral.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_neutral.setLineWidth(1)
        self.counter_neutral.setMidLineWidth(0)
        self.counter_neutral.setSmallDecimalPoint(False)
        self.counter_neutral.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_neutral.setObjectName("counter_neutral")
        self.label_cam_neutral = QtWidgets.QLabel(parent=self.contadores)
        self.label_cam_neutral.setGeometry(QtCore.QRect(460, 30, 50, 50))
        self.label_cam_neutral.setText("")
        self.label_cam_neutral.setPixmap(QtGui.QPixmap(":/imgs/imgs/Neutral face.png"))
        self.label_cam_neutral.setScaledContents(True)
        self.label_cam_neutral.setObjectName("label_cam_neutral")
        self.counter_surp = QtWidgets.QLCDNumber(parent=self.contadores)
        self.counter_surp.setGeometry(QtCore.QRect(240, 80, 64, 23))
        self.counter_surp.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_surp.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_surp.setLineWidth(1)
        self.counter_surp.setMidLineWidth(0)
        self.counter_surp.setSmallDecimalPoint(False)
        self.counter_surp.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_surp.setObjectName("counter_surp")
        self.counter_happy = QtWidgets.QLCDNumber(parent=self.contadores)
        self.counter_happy.setGeometry(QtCore.QRect(40, 80, 64, 23))
        self.counter_happy.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_happy.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_happy.setLineWidth(1)
        self.counter_happy.setMidLineWidth(0)
        self.counter_happy.setSmallDecimalPoint(False)
        self.counter_happy.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_happy.setObjectName("counter_happy")
        self.counter_sad = QtWidgets.QLCDNumber(parent=self.contadores)
        self.counter_sad.setGeometry(QtCore.QRect(140, 80, 64, 23))
        self.counter_sad.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_sad.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_sad.setLineWidth(1)
        self.counter_sad.setMidLineWidth(0)
        self.counter_sad.setSmallDecimalPoint(False)
        self.counter_sad.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_sad.setObjectName("counter_sad")
        self.counter_disgust = QtWidgets.QLCDNumber(parent=self.contadores)
        self.counter_disgust.setGeometry(QtCore.QRect(560, 80, 64, 23))
        self.counter_disgust.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_disgust.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_disgust.setLineWidth(1)
        self.counter_disgust.setMidLineWidth(0)
        self.counter_disgust.setSmallDecimalPoint(False)
        self.counter_disgust.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_disgust.setObjectName("counter_disgust")
        self.counter_anger = QtWidgets.QLCDNumber(parent=self.contadores)
        self.counter_anger.setGeometry(QtCore.QRect(670, 80, 64, 23))
        self.counter_anger.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.counter_anger.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.counter_anger.setLineWidth(1)
        self.counter_anger.setMidLineWidth(0)
        self.counter_anger.setSmallDecimalPoint(False)
        self.counter_anger.setMode(QtWidgets.QLCDNumber.Mode.Dec)
        self.counter_anger.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.counter_anger.setObjectName("counter_anger")
        self.label_cam_fear = QtWidgets.QLabel(parent=self.contadores)
        self.label_cam_fear.setGeometry(QtCore.QRect(360, 30, 50, 50))
        self.label_cam_fear.setText("")
        self.label_cam_fear.setPixmap(QtGui.QPixmap(":/imgs/imgs/Fearful face.png"))
        self.label_cam_fear.setScaledContents(True)
        self.label_cam_fear.setObjectName("label_cam_fear")
        self.label_cam_disgust = QtWidgets.QLabel(parent=self.contadores)
        self.label_cam_disgust.setGeometry(QtCore.QRect(570, 30, 50, 50))
        self.label_cam_disgust.setText("")
        self.label_cam_disgust.setPixmap(QtGui.QPixmap(":/imgs/imgs/Pensive face.png"))
        self.label_cam_disgust.setScaledContents(True)
        self.label_cam_disgust.setObjectName("label_cam_disgust")
        self.label_cam_emometer = QtWidgets.QLabel(parent=self.contadores)
        self.label_cam_emometer.setGeometry(QtCore.QRect(200, -4, 391, 41))
        self.label_cam_emometer.setScaledContents(True)
        self.label_cam_emometer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_cam_emometer.setWordWrap(False)
        self.label_cam_emometer.setObjectName("label_cam_emometer")
        self.tabWidget.addTab(self.cam_tab, "")

        self.retranslateUi(mainWindow)
        self.tabWidget.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Happy Or Not "))
        self.toolButton_image.setText(_translate("mainWindow", "..."))
        self.load_image_button.setText(_translate("mainWindow", "Play Video"))
        self.labelimage.setText(_translate("mainWindow", "Emotions Detected"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.images_tab), _translate("mainWindow", "Images"))
        self.stop_screen_button.setText(_translate("mainWindow", "Parar captura"))
        self.load_screen_button.setText(_translate("mainWindow", "Capturar"))
        self.label_screen_emometer.setText(_translate("mainWindow", "EMO METER"))
        self.labelscreen1.setText(_translate("mainWindow", "Emotions Detected"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.capture_tab), _translate("mainWindow", "Pantalla"))
        self.load_video_button.setText(_translate("mainWindow", "Play Video"))
        self.labelcaamvideo.setText(_translate("mainWindow", "Emotions Detected"))
        self.toolButton.setText(_translate("mainWindow", "..."))
        self.stop_video_button.setText(_translate("mainWindow", "Stop Video"))
        self.label_video_emometer.setText(_translate("mainWindow", "EMO METER"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.video_tab), _translate("mainWindow", "Video"))
        self.load_cam_button.setText(_translate("mainWindow", "Load Camera"))
        self.labelcaam1.setText(_translate("mainWindow", "Emotions Detected"))
        self.stop_cam_button.setText(_translate("mainWindow", "Stop Camera"))
        self.label_cam_emometer.setText(_translate("mainWindow", "EMO METER"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.cam_tab), _translate("mainWindow", "Cam"))
