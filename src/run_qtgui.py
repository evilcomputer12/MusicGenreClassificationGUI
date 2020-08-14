import numpy as np
import torch
import sys

from collections import Counter
from sklearn.preprocessing import LabelEncoder

from librosa.core import load
from librosa.feature import melspectrogram
from librosa import power_to_db

from model import genreNet
from config import MODELPATH
from config import GENRES

import os

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog

import warnings

import threading

warnings.filterwarnings("ignore")

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 10, 363, 317))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.button2 = QtWidgets.QPushButton(self.widget)
        self.button2.setObjectName("button2")
        self.gridLayout.addWidget(self.button2, 4, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.button1 = QtWidgets.QPushButton(self.widget)
        self.button1.setObjectName("button1")
        self.gridLayout.addWidget(self.button1, 3, 0, 1, 1)
        self.openfile = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.openfile.setFont(font)
        self.openfile.setAlignment(QtCore.Qt.AlignCenter)
        self.openfile.setObjectName("openfile")
        self.gridLayout.addWidget(self.openfile, 1, 0, 1, 1)
        self.text = QtWidgets.QPlainTextEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.text.setFont(font)
        self.text.setObjectName("text")
        # self.text.setReadOnly(True)
        self.gridLayout.addWidget(self.text, 6, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button2.setText(_translate("MainWindow", "Откри жанр"))
        self.label.setText(_translate("MainWindow", "Отвори музичка датотека и откри го нејзиниот музички жанр"))
        self.button1.setText(_translate("MainWindow", "Отвори датотека"))
        self.button1.clicked.connect(self.openFile)
        self.button2.clicked.connect(self.thread)
        self.text.setReadOnly(True)
        self.button2.setEnabled(False)
        self.pbar = QtWidgets.QProgressBar(self.widget)
        self.pbar.setTextVisible(False)
        self.pbar.setObjectName("pbar")
        self.gridLayout.addWidget(self.pbar, 5, 0, 1, 1)
        self.pbar.setMinimum(0)
        self.pbar.setMaximum(100)


    def openFile(self):
        options = QtWidgets.QFileDialog.Options()
        global fileName
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open Music Files",
            os.getcwd(),
            "Music Files (*.mp3);;All Files (*)",
            options=options)
        if len(fileName) != 0:
            self.openfile.setText(os.path.basename(fileName))
            self.text.clear()
            self.button2.setEnabled(True)
        else:
            self.openfile.clear()
            self.button2.setEnabled(False)

    def main(self):
        le = LabelEncoder().fit(GENRES)
        # ------------------------------- #
        ## LOAD TRAINED GENRENET MODEL
        net = genreNet()
        net.load_state_dict(torch.load(MODELPATH, map_location='cpu'))
        # ------------------------------- #
        ## LOAD AUDIO
        audio_path = fileName
        # os.path.basename()
        y, sr = load(audio_path, mono=True, sr=22050)
        # ------------------------------- #
        ## GET CHUNKS OF AUDIO SPECTROGRAMS
        S = melspectrogram(y, sr).T
        S = S[:-1 * (S.shape[0] % 128)]
        num_chunk = S.shape[0] / 128
        data_chunks = np.split(S, num_chunk)
        # ------------------------------- #
        ## CLASSIFY SPECTROGRAMS
        genres = list()
        for i, data in enumerate(data_chunks):
            data = torch.FloatTensor(data).view(1, 1, 128, 128)
            preds = net(data)
            pred_val, pred_index = preds.max(1)
            pred_index = pred_index.data.numpy()
            pred_val = np.exp(pred_val.data.numpy()[0])
            pred_genre = le.inverse_transform(pred_index).item()
            if pred_val >= 0.5:
                genres.append(pred_genre)
        # ------------------------------- #
        s = float(sum([v for k, v in dict(Counter(genres)).items()]))
        pos_genre = sorted([(k, v / s * 100) for k, v in dict(Counter(genres)).items()], key=lambda x: x[1],
                           reverse=True)
        for genre, pos in pos_genre:
            print("%10s: \t%.2f\t%%\n" % (genre, pos))
            self.text.insertPlainText("%10s: \t%.2f\t%%\n" % (genre, pos))
            self.button1.setEnabled(True)
            self.button2.setEnabled(True)
            self.pbar.setMinimum(0)
            self.pbar.setMaximum(100)
            #self.pbar.deleteLater()
        return
    def thread(self):
        self.button1.setEnabled(False)
        self.button2.setEnabled(False)
        self.pbar.setMinimum(0)
        self.pbar.setMaximum(0)
        t=threading.Thread(target=self.main, args=())
        t.start()
        return

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())