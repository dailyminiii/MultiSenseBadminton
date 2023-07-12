import sys
import os
import h5py
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def timeSearch(dataset, time):
    left = 0
    right = dataset.shape[0]
    half = int(dataset.shape[0] / 2)
    while True:
        if time >= dataset[half] and time <= dataset[half+1]:
            return half
        elif time > dataset[half]:
            left = half
        else:
            right = half

        half = int((left + right) / 2)

class Visualizer(QWidget):

    def __init__(self):
        super().__init__()
        self.setFixedSize(1600, 900)

        self.initUI()

    def initUI(self):

        # 첫 번째 행
        self.hbox1 = QVBoxLayout()
        self.hbox2 = QVBoxLayout()
        self.hbox3 = QHBoxLayout()

        # 왼쪽 위젯
        vbox1 = QHBoxLayout()
        vbox2 = QHBoxLayout()
        vbox3 = QHBoxLayout()

        label1 = QLabel('P1       Select a Subject Number:', self)
        label1.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        label2 = QLabel('Select a Stroke Type:', self)
        label2.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        label3 = QLabel('Select a Stroke Number:', self)
        label3.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        label4 = QLabel('P2       Select a Subject Number:', self)
        label4.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        label5 = QLabel('Select a Stroke Type:', self)
        label5.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        label6 = QLabel('Select a Stroke Number:', self)
        label6.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")

        self.over1 = 0
        self.back1 = 0
        self.over2 = 0
        self.back2 = 0

        self.file_list1 = []
        self.file_list2 = []
        self.filename1 = ''
        self.filename2 = ''
        self.filename3 = ''
        self.filename4 = ''

        # 버튼 생성
        self.button1 = QPushButton('EMG LEG', self)
        self.button2 = QPushButton('EMG UpperArm', self)
        self.button3 = QPushButton('EMG LowerArm', self)
        self.button4 = QPushButton('Insole', self)
        self.button5 = QPushButton('Gaze', self)
        self.button6 = QPushButton('Skeleton', self)
        #self.button7 = QPushButton('Button 7', self)

        self.sub1 = QComboBox(self)
        self.sub2 = QComboBox(self)
        self.sub3 = QComboBox(self)
        self.sub4 = QComboBox(self)
        self.sub5 = QComboBox(self)
        self.sub6 = QComboBox(self)

        self.sub1.addItems(["Select"])
        self.sub4.addItems(["Select"])
        for i in range(29):
            self.sub1.addItems(['S{0:02d}'.format(i)])
            self.sub4.addItems(['S{0:02d}'.format(i)])

        self.sub1.activated[str].connect(self.onActivated1)
        self.sub4.activated[str].connect(self.onActivated2)

        self.sub2.addItems(["Overhead Clear", "Backhand Drive"])
        self.sub5.addItems(["Overhead Clear", "Backhand Drive"])

        self.sub2.activated[str].connect(self.onActivated_action1)
        self.sub5.activated[str].connect(self.onActivated_action2)

        self.button1.clicked.connect(self.on_click_EMG)
        self.button2.clicked.connect(self.on_click_EMG_UpperArm)
        self.button3.clicked.connect(self.on_click_EMG_LowerArm)
        self.button4.clicked.connect(self.on_click_Insole)
        self.button5.clicked.connect(self.on_click_Eye)
        self.button6.clicked.connect(self.on_click_PNS)
        #self.button7.clicked.connect(self.on_click)

        # 왼쪽 위젯에 버튼과 ComboBox 추가
        vbox1.addWidget(label1)
        vbox1.addWidget(self.sub1)
        vbox1.addWidget(label2)
        vbox1.addWidget(self.sub2)
        vbox1.addWidget(label3)
        vbox1.addWidget(self.sub3)
        vbox2.addWidget(label4)
        vbox2.addWidget(self.sub4)
        vbox2.addWidget(label5)
        vbox2.addWidget(self.sub5)
        vbox2.addWidget(label6)
        vbox2.addWidget(self.sub6)
        vbox3.addWidget(self.button1)
        vbox3.addWidget(self.button2)
        vbox3.addWidget(self.button3)
        vbox3.addWidget(self.button4)
        vbox3.addWidget(self.button5)
        vbox3.addWidget(self.button6)
        #vbox3.addWidget(self.button7)

        self.hbox1.addLayout(vbox1)
        self.hbox1.addLayout(vbox2)
        self.hbox1.addLayout(vbox3)

        self.vbox2 = QVBoxLayout()
        self.vbox2.addWidget(QWidget())
        self.hbox3.addLayout(self.vbox2)

        self.vbox3 = QVBoxLayout()
        self.vbox3.addWidget(QWidget())
        self.hbox3.addLayout(self.vbox3)

        self.hbox2.addLayout(self.hbox3)
        self.hbox1.addLayout(self.hbox2)

        self.setLayout(self.hbox1)

        self.center()
        self.setWindowTitle('Visualizer')
        self.show()

        self.pen = pg.mkPen(color=(255, 0, 0))
        self.splited_data1 = 0
        self.update_count1 = 0
        self.timer1 = QTimer(self)

        self.splited_data2 = 0
        self.update_count2 = 0
        self.timer2 = QTimer(self)

        self._segment_labels = [
            'Hips',
            'Right UpLeg',
            'Right Leg',
            'Right Foot',
            'Left UpLeg',
            'Left Leg',
            'Left Foot',
            'Spine',
            'Spine1',
            'Spine2',
            'Neck',
            'Neck1',
            'Head',
            'Right Shoulder',
            'Right Arm',
            'Right ForeArm',
            'Right Hand',
            'Left Shoulder',
            'Left Arm',
            'Left ForeArm',
            'Left Hand',
        ]

        self._segment_chains_labels_toPlot = {

            'Left Legs': ['Hips', 'Left UpLeg', 'Left Leg', 'Left Foot'],
            'Right Legs': ['Hips', 'Right UpLeg', 'Right Leg', 'Right Foot'],
            'Spines': ['Head', 'Neck1', 'Neck1', 'Spine2', 'Spine1', 'Spine', 'Hips', ],
            'Shoulders': ['Left Shoulder', 'Neck', 'Right Shoulder'],
            'Left Arms': ['Left Shoulder', 'Left Arm', 'Left ForeArm', 'Left Hand'],
            'Right Arms': ['Right Shoulder', 'Right Arm', 'Right ForeArm', 'Right Hand'],
        }
        self._segment_chains_indexes_toPlot = dict()
        for (chain_name, chain_labels) in self._segment_chains_labels_toPlot.items():
            # print(chain_name + str(chain_labels))
            segment_indexes = []
            for chain_label in chain_labels:
                segment_indexes.append(self._segment_labels.index(chain_label))
            self._segment_chains_indexes_toPlot[chain_name] = segment_indexes

        segment_positions_cm = np.zeros([len(self._segment_labels), 3])

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def onActivated1(self, text):
        files = os.listdir()
        self.file_list1 = []
        for file in files:
            if text in file:
                self.file_list1.append(file)

        if len(self.file_list1) == 1:
            self.filename1 = self.file_list1[0]
            self.over1, self.back1 = self.searchActivities(self.filename1)
        else :
            with h5py.File(self.file_list1[0], "r") as f:
                dataset = f['experiment-activities']
                data = dataset['activities']['data']
                if b'Backhand Driving' in data[0].tolist()[0]:
                    self.filename3 = self.file_list1[0]
                    self.filename1 = self.file_list1[1]
                    self.over1, _ = self.searchActivities(self.filename1)
                    _, self.back1 = self.searchActivities(self.filename3)
                else :
                    self.filename1 = self.file_list1[0]
                    self.filename3 = self.file_list1[1]
                    self.over1, _ = self.searchActivities(self.filename1)
                    _, self.back1 = self.searchActivities(self.filename3)

        self.sub3.clear()
        if self.sub2.currentText() == "Overhead Clear":
            for i in range(1, self.over1+1):
                self.sub3.addItems([str(i)])
        else :
            for i in range(1, self.back1+1):
                self.sub3.addItems([str(i)])

    def onActivated2(self, text):
        files = os.listdir()
        self.file_list2 = []
        for file in files:
            if text in file:
                self.file_list2.append(file)

        if len(self.file_list2) == 1:
            self.filename2 = self.file_list2[0]
            self.over2, self.back2 = self.searchActivities(self.filename2)
        else :
            with h5py.File(self.file_list2[0], "r") as f:
                dataset = f['experiment-activities']
                data = dataset['activities']['data']
                if b'Backhand Driving' in data[0].tolist()[0]:
                    self.filename4 = self.file_list2[0]
                    self.filename2 = self.file_list2[1]
                    self.over2, _ = self.searchActivities(self.filename2)
                    _, self.back2 = self.searchActivities(self.filename4)
                else :
                    self.filename2 = self.file_list2[0]
                    self.filename4 = self.file_list2[1]
                    self.over2, _ = self.searchActivities(self.filename2)
                    _, self.back2 = self.searchActivities(self.filename4)


        self.sub6.clear()
        if self.sub5.currentText() == "Overhead Clear":
            for i in range(1, self.over2+1):
                self.sub6.addItems([str(i)])
        else :
            for i in range(1, self.back2+1):
                self.sub6.addItems([str(i)])

    def onActivated_action1(self):
        self.sub3.clear()
        if self.sub2.currentText() == "Overhead Clear":
            for i in range(1, self.over1+1):
                self.sub3.addItems([str(i)])
        else :
            for i in range(1, self.back1+1):
                self.sub3.addItems([str(i)])

    def onActivated_action2(self):
        self.sub6.clear()
        if self.sub5.currentText() == "Overhead Clear":
            for i in range(1, self.over2+1):
                self.sub6.addItems([str(i)])
        else :
            for i in range(1, self.back2+1):
                self.sub6.addItems([str(i)])

    def update_graph1(self):
        if self.update_count1 < len(self.splited_data1):
            self.EMG_DATA1 = np.concatenate([self.EMG_DATA1, self.splited_data1[self.update_count1][:,0]])
            self.EMG_DATA2 = np.concatenate([self.EMG_DATA2, self.splited_data1[self.update_count1][:,1]])
            self.EMG_DATA3 = np.concatenate([self.EMG_DATA3, self.splited_data1[self.update_count1][:,2]])
            self.EMG_DATA4 = np.concatenate([self.EMG_DATA4, self.splited_data1[self.update_count1][:,3]])
            self.update_count1 += 1
            self.emg_plot1.setData(self.EMG_DATA1)
            self.emg_plot2.setData(self.EMG_DATA2)
            self.emg_plot3.setData(self.EMG_DATA3)
            self.emg_plot4.setData(self.EMG_DATA4)
        else:
            self.splited_data1 = []
            self.timer1.stop()
            self.timer1.deleteLater()
            self.timer1 = QTimer(self)
            self.EMG_DATA1 = np.array([], dtype=np.float32)
            self.EMG_DATA2 = np.array([], dtype=np.float32)
            self.EMG_DATA3 = np.array([], dtype=np.float32)
            self.EMG_DATA4 = np.array([], dtype=np.float32)
            self.update_count1 = 0

    def update_graph2(self):
        if self.update_count2 < len(self.splited_data2):
            self.EMG_DATA5 = np.concatenate([self.EMG_DATA5, self.splited_data2[self.update_count2][:,0]])
            self.EMG_DATA6 = np.concatenate([self.EMG_DATA6, self.splited_data2[self.update_count2][:,1]])
            self.EMG_DATA7 = np.concatenate([self.EMG_DATA7, self.splited_data2[self.update_count2][:,2]])
            self.EMG_DATA8 = np.concatenate([self.EMG_DATA8, self.splited_data2[self.update_count2][:,3]])
            self.update_count2 += 1
            self.emg_plot5.setData(self.EMG_DATA5)
            self.emg_plot6.setData(self.EMG_DATA6)
            self.emg_plot7.setData(self.EMG_DATA7)
            self.emg_plot8.setData(self.EMG_DATA8)
        else:
            self.splited_data2 = []
            self.timer2.stop()
            self.timer2.deleteLater()
            self.timer2 = QTimer(self)
            self.EMG_DATA5 = np.array([], dtype=np.float32)
            self.EMG_DATA6 = np.array([], dtype=np.float32)
            self.EMG_DATA7 = np.array([], dtype=np.float32)
            self.EMG_DATA8 = np.array([], dtype=np.float32)
            self.update_count2 = 0

    def update_upper_graph1(self):
        if self.update_count1 < len(self.splited_data1):
            self.EMG_DATA1 = np.concatenate([self.EMG_DATA1, self.splited_data1[self.update_count1][:,0]])
            self.EMG_DATA2 = np.concatenate([self.EMG_DATA2, self.splited_data1[self.update_count1][:,1]])
            self.EMG_DATA3 = np.concatenate([self.EMG_DATA3, self.splited_data1[self.update_count1][:,2]])
            self.EMG_DATA4 = np.concatenate([self.EMG_DATA4, self.splited_data1[self.update_count1][:,3]])
            self.EMG_DATA5 = np.concatenate([self.EMG_DATA5, self.splited_data1[self.update_count1][:,4]])
            self.EMG_DATA6 = np.concatenate([self.EMG_DATA6, self.splited_data1[self.update_count1][:,5]])
            self.EMG_DATA7 = np.concatenate([self.EMG_DATA7, self.splited_data1[self.update_count1][:,6]])
            self.EMG_DATA8 = np.concatenate([self.EMG_DATA8, self.splited_data1[self.update_count1][:,7]])
            self.update_count1 += 1
            self.emg_plot1.setData(self.EMG_DATA1)
            self.emg_plot2.setData(self.EMG_DATA2)
            self.emg_plot3.setData(self.EMG_DATA3)
            self.emg_plot4.setData(self.EMG_DATA4)
            self.emg_plot5.setData(self.EMG_DATA5)
            self.emg_plot6.setData(self.EMG_DATA6)
            self.emg_plot7.setData(self.EMG_DATA7)
            self.emg_plot8.setData(self.EMG_DATA8)
        else:
            self.splited_data1 = []
            self.timer1.stop()
            self.timer1.deleteLater()
            self.timer1 = QTimer(self)
            self.EMG_DATA1 = np.array([], dtype=np.float32)
            self.EMG_DATA2 = np.array([], dtype=np.float32)
            self.EMG_DATA3 = np.array([], dtype=np.float32)
            self.EMG_DATA4 = np.array([], dtype=np.float32)
            self.EMG_DATA5 = np.array([], dtype=np.float32)
            self.EMG_DATA6 = np.array([], dtype=np.float32)
            self.EMG_DATA7 = np.array([], dtype=np.float32)
            self.EMG_DATA8 = np.array([], dtype=np.float32)
            self.update_count1 = 0

    def update_upper_graph2(self):
        if self.update_count2 < len(self.splited_data2):
            self.EMG_DATA9 = np.concatenate([self.EMG_DATA9, self.splited_data2[self.update_count2][:,0]])
            self.EMG_DATA10 = np.concatenate([self.EMG_DATA10, self.splited_data2[self.update_count2][:,1]])
            self.EMG_DATA11 = np.concatenate([self.EMG_DATA11, self.splited_data2[self.update_count2][:,2]])
            self.EMG_DATA12 = np.concatenate([self.EMG_DATA12, self.splited_data2[self.update_count2][:,3]])
            self.EMG_DATA13 = np.concatenate([self.EMG_DATA13, self.splited_data2[self.update_count2][:,4]])
            self.EMG_DATA14 = np.concatenate([self.EMG_DATA14, self.splited_data2[self.update_count2][:,5]])
            self.EMG_DATA15 = np.concatenate([self.EMG_DATA15, self.splited_data2[self.update_count2][:,6]])
            self.EMG_DATA16 = np.concatenate([self.EMG_DATA16, self.splited_data2[self.update_count2][:,7]])
            self.update_count2 += 1
            self.emg_plot9.setData(self.EMG_DATA9)
            self.emg_plot10.setData(self.EMG_DATA10)
            self.emg_plot11.setData(self.EMG_DATA11)
            self.emg_plot12.setData(self.EMG_DATA12)
            self.emg_plot13.setData(self.EMG_DATA13)
            self.emg_plot14.setData(self.EMG_DATA14)
            self.emg_plot15.setData(self.EMG_DATA15)
            self.emg_plot16.setData(self.EMG_DATA16)
        else:
            self.splited_data2 = []
            self.timer2.stop()
            self.timer2.deleteLater()
            self.timer2 = QTimer(self)
            self.EMG_DATA9 = np.array([], dtype=np.float32)
            self.EMG_DATA10 = np.array([], dtype=np.float32)
            self.EMG_DATA11 = np.array([], dtype=np.float32)
            self.EMG_DATA12 = np.array([], dtype=np.float32)
            self.EMG_DATA13 = np.array([], dtype=np.float32)
            self.EMG_DATA14 = np.array([], dtype=np.float32)
            self.EMG_DATA15 = np.array([], dtype=np.float32)
            self.EMG_DATA16 = np.array([], dtype=np.float32)
            self.update_count2 = 0

    def update_eye_graph1(self):
        if self.update_count1 < len(self.splited_data1):
            self.EMG_DATA1 = np.concatenate([self.EMG_DATA1, self.splited_data1[self.update_count1][:,0]])
            self.EMG_DATA2 = np.concatenate([self.EMG_DATA2, self.splited_data1[self.update_count1][:,1]])
            self.update_count1 += 1
            self.emg_plot1.setData(self.EMG_DATA1)
            self.emg_plot2.setData(self.EMG_DATA2)
        else:
            self.splited_data1 = []
            self.timer1.stop()
            self.timer1.deleteLater()
            self.timer1 = QTimer(self)
            self.EMG_DATA1 = np.array([], dtype=np.float32)
            self.EMG_DATA2 = np.array([], dtype=np.float32)
            self.update_count1 = 0

    def update_eye_graph2(self):
        if self.update_count2 < len(self.splited_data2):
            self.EMG_DATA3 = np.concatenate([self.EMG_DATA3, self.splited_data2[self.update_count2][:,0]])
            self.EMG_DATA4 = np.concatenate([self.EMG_DATA4, self.splited_data2[self.update_count2][:,1]])
            self.update_count2 += 1
            self.emg_plot3.setData(self.EMG_DATA3)
            self.emg_plot4.setData(self.EMG_DATA4)
        else:
            self.splited_data2 = []
            self.timer2.stop()
            self.timer2.deleteLater()
            self.timer2 = QTimer(self)
            self.EMG_DATA3 = np.array([], dtype=np.float32)
            self.EMG_DATA4 = np.array([], dtype=np.float32)
            self.update_count2 = 0

    def update_insole_graph1(self):
        if self.update_count1 < len(self.splited_data1):
            self.heatmap1.set_data(self.splited_data1[self.update_count1][-1,:].reshape(1,16))
            self.fig1.canvas.draw_idle()
            self.heatmap2.set_data(self.splited_data2[self.update_count1][-1,:].reshape(1,16))
            self.fig2.canvas.draw_idle()
            self.update_count1 += 1
        else:
            self.splited_data1 = []
            self.splited_data2 = []
            self.timer1.stop()
            self.timer1.deleteLater()
            self.timer1 = QTimer(self)
            self.update_count1 = 0

    def update_insole_graph2(self):
        if self.update_count2 < len(self.splited_data3):
            self.heatmap3.set_data(self.splited_data3[self.update_count2][-1,:].reshape(1,16))
            self.fig3.canvas.draw_idle()
            self.heatmap4.set_data(self.splited_data4[self.update_count2][-1,:].reshape(1,16))
            self.fig4.canvas.draw_idle()
            self.update_count2 += 1

        else:
            self.splited_data3 = []
            self.splited_data4 = []
            self.timer2.stop()
            self.timer2.deleteLater()
            self.timer2 = QTimer(self)
            self.update_count2 = 0

    def update_pns_graph1(self):
        if self.update_count1 < len(self.splited_data1):
            joint_cm = self.splited_data1[self.update_count1][-1, :].reshape(21, 3)
            self.ax1.clear()

            for (chain_index, chain_name) in enumerate(self._segment_chains_indexes_toPlot.keys()):
                segment_indexes = self._segment_chains_indexes_toPlot[chain_name]
                segment_xyz_cm = joint_cm[segment_indexes, :]
                plot_x = segment_xyz_cm[:, 2]
                plot_y = segment_xyz_cm[:, 0]
                plot_z = segment_xyz_cm[:, 1]
                self.ax1.plot(plot_x, plot_y, plot_z, 'r-o', markersize=5)

            self.ax1.set_xlim([self.z1 - 75, self.z1 + 75])
            self.ax1.set_ylim([self.x1 - 75, self.x1 + 75])
            self.ax1.set_zlim([0, 200])

            self.fig1.canvas.draw()
            self.fig1.canvas.flush_events()
            self.update_count1 += 1
        else:
            self.splited_data1 = []
            self.timer1.stop()
            self.timer1.deleteLater()
            self.timer1 = QTimer(self)
            self.update_count1 = 0

    def update_pns_graph2(self):
        if self.update_count2 < len(self.splited_data2):
            joint_cm = self.splited_data2[self.update_count2][-1, :].reshape(21, 3)
            self.ax2.clear()

            for (chain_index, chain_name) in enumerate(self._segment_chains_indexes_toPlot.keys()):
                segment_indexes = self._segment_chains_indexes_toPlot[chain_name]
                segment_xyz_cm = joint_cm[segment_indexes, :]
                plot_x = segment_xyz_cm[:, 2]
                plot_y = segment_xyz_cm[:, 0]
                plot_z = segment_xyz_cm[:, 1]
                self.ax2.plot(plot_x, plot_y, plot_z, 'r-o', markersize=5)

            self.ax2.set_xlim([self.z2 - 75, self.z2 + 75])
            self.ax2.set_ylim([self.x2 - 75, self.x2 + 75])
            self.ax2.set_zlim([0, 200])

            self.fig2.canvas.draw()
            self.fig2.canvas.flush_events()
            self.update_count2 += 1
        else:
            self.splited_data2 = []
            self.timer2.stop()
            self.timer2.deleteLater()
            self.timer2 = QTimer(self)
            self.update_count2 = 0

    def on_click_EMG(self):
        self.splited_data1 = []
        self.timer1.stop()
        self.timer1.deleteLater()
        self.timer1 = QTimer(self)
        self.update_count1 = 0
        self.splited_data2 = []
        self.timer2.stop()
        self.timer2.deleteLater()
        self.timer2 = QTimer(self)
        self.update_count2 = 0
        self.on_click_EMG1()
        self.on_click_EMG2()

    def on_click_EMG1(self):

        if len(self.file_list1) == 1:
            filename = self.filename1
            if self.sub2.currentText() == "Overhead Clear":
                index_start = (int(self.sub3.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub3.currentText()) - 1) * 2 + self.over1 * 2
        else :
            index_start = (int(self.sub3.currentText()) - 1) * 2
            if self.sub2.currentText() == "Overhead Clear":
                filename = self.filename1
            else:
                filename = self.filename3

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            EMG = f['cgx-aim-leg-emg']
            EMG_data = EMG['emg-values']['time_str']

            data_index_start = timeSearch(EMG_data, time1)
            data_index_stop = timeSearch(EMG_data, time2)

            searched_data = EMG['emg-values']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data1 = np.array_split(searched_data, split_index)

        self.y = searched_data[:,0]

        self.clearWidget(self.vbox2)

        self.vbox2.addWidget(QWidget())

        label = QLabel('P1 Leg EMG')
        label.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        self.vbox2.layout().addWidget(label)

        self.graphWidget1 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget1)
        self.graphWidget1.setBackground('w')

        self.graphWidget2 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget2)
        self.graphWidget2.setBackground('w')

        self.graphWidget3 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget3)
        self.graphWidget3.setBackground('w')

        self.graphWidget4 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget4)
        self.graphWidget4.setBackground('w')

        self.EMG_DATA1 = np.array([], dtype=np.float32)
        self.EMG_DATA2 = np.array([], dtype=np.float32)
        self.EMG_DATA3 = np.array([], dtype=np.float32)
        self.EMG_DATA4 = np.array([], dtype=np.float32)
        self.emg_plot1 = self.graphWidget1.plot(pen=self.pen)
        self.emg_plot2 = self.graphWidget2.plot(pen=self.pen)
        self.emg_plot3 = self.graphWidget3.plot(pen=self.pen)
        self.emg_plot4 = self.graphWidget4.plot(pen=self.pen)

        self.timer1.timeout.connect(self.update_graph1)
        self.timer1.start(50)

    def on_click_EMG2(self):

        if len(self.file_list2) == 1:
            filename = self.filename2
            if self.sub5.currentText() == "Overhead Clear":
                index_start = (int(self.sub6.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub6.currentText()) - 1) * 2 + self.over2 * 2
        else :
            index_start = (int(self.sub6.currentText()) - 1) * 2
            if self.sub5.currentText() == "Overhead Clear":
                filename = self.filename2
            else:
                filename = self.filename4

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            EMG = f['cgx-aim-leg-emg']
            EMG_data = EMG['emg-values']['time_str']

            data_index_start = timeSearch(EMG_data, time1)
            data_index_stop = timeSearch(EMG_data, time2)

            searched_data = EMG['emg-values']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data2 = np.array_split(searched_data, split_index)

        self.y = searched_data[:,0]

        self.clearWidget(self.vbox3)

        self.vbox3.addWidget(QWidget())

        label = QLabel('P2 Leg EMG')
        label.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        self.vbox3.layout().addWidget(label)

        self.graphWidget5 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget5)
        self.graphWidget5.setBackground('w')

        self.graphWidget6 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget6)
        self.graphWidget6.setBackground('w')

        self.graphWidget7 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget7)
        self.graphWidget7.setBackground('w')

        self.graphWidget8 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget8)
        self.graphWidget8.setBackground('w')

        self.EMG_DATA5 = np.array([], dtype=np.float32)
        self.EMG_DATA6 = np.array([], dtype=np.float32)
        self.EMG_DATA7 = np.array([], dtype=np.float32)
        self.EMG_DATA8 = np.array([], dtype=np.float32)
        self.emg_plot5 = self.graphWidget5.plot(pen=self.pen)
        self.emg_plot6 = self.graphWidget6.plot(pen=self.pen)
        self.emg_plot7 = self.graphWidget7.plot(pen=self.pen)
        self.emg_plot8 = self.graphWidget8.plot(pen=self.pen)

        self.timer2.timeout.connect(self.update_graph2)
        self.timer2.start(50)

    def on_click_EMG_UpperArm(self):
        self.splited_data1 = []
        self.timer1.stop()
        self.timer1.deleteLater()
        self.timer1 = QTimer(self)
        self.update_count1 = 0
        self.splited_data2 = []
        self.timer2.stop()
        self.timer2.deleteLater()
        self.timer2 = QTimer(self)
        self.update_count2 = 0
        self.on_click_EMG1_Upper()
        self.on_click_EMG2_Upper()

    def on_click_EMG1_Upper(self):

        if len(self.file_list1) == 1:
            filename = self.filename1
            if self.sub2.currentText() == "Overhead Clear":
                index_start = (int(self.sub3.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub3.currentText()) - 1) * 2 + self.over1 * 2
        else :
            index_start = (int(self.sub3.currentText()) - 1) * 2
            if self.sub2.currentText() == "Overhead Clear":
                filename = self.filename1
            else:
                filename = self.filename3

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            EMG = f['gforce-upperarm-emg']
            EMG_data = EMG['emg-values']['time_str']

            data_index_start = timeSearch(EMG_data, time1)
            data_index_stop = timeSearch(EMG_data, time2)

            searched_data = EMG['emg-values']['data'][data_index_start:data_index_stop]
            print(time1, time2)
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data1 = np.array_split(searched_data, split_index)

        self.clearWidget(self.vbox2)

        self.vbox2.addWidget(QWidget())

        label = QLabel('P1 UpperArm EMG')
        label.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        self.vbox2.layout().addWidget(label)

        self.graphWidget1 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget1)
        self.graphWidget1.setBackground('w')

        self.graphWidget2 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget2)
        self.graphWidget2.setBackground('w')

        self.graphWidget3 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget3)
        self.graphWidget3.setBackground('w')

        self.graphWidget4 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget4)
        self.graphWidget4.setBackground('w')

        self.graphWidget5 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget5)
        self.graphWidget5.setBackground('w')

        self.graphWidget6 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget6)
        self.graphWidget6.setBackground('w')

        self.graphWidget7 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget7)
        self.graphWidget7.setBackground('w')

        self.graphWidget8 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget8)
        self.graphWidget8.setBackground('w')

        self.EMG_DATA1 = np.array([], dtype=np.float32)
        self.EMG_DATA2 = np.array([], dtype=np.float32)
        self.EMG_DATA3 = np.array([], dtype=np.float32)
        self.EMG_DATA4 = np.array([], dtype=np.float32)
        self.EMG_DATA5 = np.array([], dtype=np.float32)
        self.EMG_DATA6 = np.array([], dtype=np.float32)
        self.EMG_DATA7 = np.array([], dtype=np.float32)
        self.EMG_DATA8 = np.array([], dtype=np.float32)
        self.emg_plot1 = self.graphWidget1.plot(pen=self.pen)
        self.emg_plot2 = self.graphWidget2.plot(pen=self.pen)
        self.emg_plot3 = self.graphWidget3.plot(pen=self.pen)
        self.emg_plot4 = self.graphWidget4.plot(pen=self.pen)
        self.emg_plot5 = self.graphWidget5.plot(pen=self.pen)
        self.emg_plot6 = self.graphWidget6.plot(pen=self.pen)
        self.emg_plot7 = self.graphWidget7.plot(pen=self.pen)
        self.emg_plot8 = self.graphWidget8.plot(pen=self.pen)

        self.timer1.timeout.connect(self.update_upper_graph1)
        self.timer1.start(50)

    def on_click_EMG2_Upper(self):

        if len(self.file_list2) == 1:
            filename = self.filename2
            if self.sub5.currentText() == "Overhead Clear":
                index_start = (int(self.sub6.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub6.currentText()) - 1) * 2 + self.over2 * 2
        else :
            index_start = (int(self.sub6.currentText()) - 1) * 2
            if self.sub5.currentText() == "Overhead Clear":
                filename = self.filename2
            else:
                filename = self.filename4

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            EMG = f['gforce-upperarm-emg']
            EMG_data = EMG['emg-values']['time_str']

            data_index_start = timeSearch(EMG_data, time1)
            data_index_stop = timeSearch(EMG_data, time2)

            searched_data = EMG['emg-values']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data2 = np.array_split(searched_data, split_index)

        self.clearWidget(self.vbox3)

        self.vbox3.addWidget(QWidget())

        label = QLabel('P2 UpperArm EMG')
        label.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        self.vbox3.layout().addWidget(label)

        self.graphWidget9 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget9)
        self.graphWidget9.setBackground('w')

        self.graphWidget10 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget10)
        self.graphWidget10.setBackground('w')

        self.graphWidget11 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget11)
        self.graphWidget11.setBackground('w')

        self.graphWidget12 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget12)
        self.graphWidget12.setBackground('w')

        self.graphWidget13 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget13)
        self.graphWidget13.setBackground('w')

        self.graphWidget14 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget14)
        self.graphWidget14.setBackground('w')

        self.graphWidget15 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget15)
        self.graphWidget15.setBackground('w')

        self.graphWidget16 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget16)
        self.graphWidget16.setBackground('w')

        self.EMG_DATA9 = np.array([], dtype=np.float32)
        self.EMG_DATA10 = np.array([], dtype=np.float32)
        self.EMG_DATA11 = np.array([], dtype=np.float32)
        self.EMG_DATA12 = np.array([], dtype=np.float32)
        self.EMG_DATA13 = np.array([], dtype=np.float32)
        self.EMG_DATA14 = np.array([], dtype=np.float32)
        self.EMG_DATA15 = np.array([], dtype=np.float32)
        self.EMG_DATA16 = np.array([], dtype=np.float32)
        self.emg_plot9 = self.graphWidget9.plot(pen=self.pen)
        self.emg_plot10 = self.graphWidget10.plot(pen=self.pen)
        self.emg_plot11 = self.graphWidget11.plot(pen=self.pen)
        self.emg_plot12 = self.graphWidget12.plot(pen=self.pen)
        self.emg_plot13 = self.graphWidget13.plot(pen=self.pen)
        self.emg_plot14 = self.graphWidget14.plot(pen=self.pen)
        self.emg_plot15 = self.graphWidget15.plot(pen=self.pen)
        self.emg_plot16 = self.graphWidget16.plot(pen=self.pen)

        self.timer2.timeout.connect(self.update_upper_graph2)
        self.timer2.start(50)

    def on_click_EMG_LowerArm(self):
        self.splited_data1 = []
        self.timer1.stop()
        self.timer1.deleteLater()
        self.timer1 = QTimer(self)
        self.update_count1 = 0
        self.splited_data2 = []
        self.timer2.stop()
        self.timer2.deleteLater()
        self.timer2 = QTimer(self)
        self.update_count2 = 0
        self.on_click_EMG1_Lower()
        self.on_click_EMG2_Lower()

    def on_click_EMG1_Lower(self):

        if len(self.file_list1) == 1:
            filename = self.filename1
            if self.sub2.currentText() == "Overhead Clear":
                index_start = (int(self.sub3.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub3.currentText()) - 1) * 2 + self.over1 * 2
        else :
            index_start = (int(self.sub3.currentText()) - 1) * 2
            if self.sub2.currentText() == "Overhead Clear":
                filename = self.filename1
            else:
                filename = self.filename3

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            EMG = f['gforce-lowerarm-emg']
            EMG_data = EMG['emg-values']['time_str']

            data_index_start = timeSearch(EMG_data, time1)
            data_index_stop = timeSearch(EMG_data, time2)

            searched_data = EMG['emg-values']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data1 = np.array_split(searched_data, split_index)

        self.clearWidget(self.vbox2)

        self.vbox2.addWidget(QWidget())

        label = QLabel('P1 Lowerarm EMG')
        label.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        self.vbox2.layout().addWidget(label)

        self.graphWidget1 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget1)
        self.graphWidget1.setBackground('w')

        self.graphWidget2 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget2)
        self.graphWidget2.setBackground('w')

        self.graphWidget3 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget3)
        self.graphWidget3.setBackground('w')

        self.graphWidget4 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget4)
        self.graphWidget4.setBackground('w')

        self.graphWidget5 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget5)
        self.graphWidget5.setBackground('w')

        self.graphWidget6 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget6)
        self.graphWidget6.setBackground('w')

        self.graphWidget7 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget7)
        self.graphWidget7.setBackground('w')

        self.graphWidget8 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget8)
        self.graphWidget8.setBackground('w')

        self.EMG_DATA1 = np.array([], dtype=np.float32)
        self.EMG_DATA2 = np.array([], dtype=np.float32)
        self.EMG_DATA3 = np.array([], dtype=np.float32)
        self.EMG_DATA4 = np.array([], dtype=np.float32)
        self.EMG_DATA5 = np.array([], dtype=np.float32)
        self.EMG_DATA6 = np.array([], dtype=np.float32)
        self.EMG_DATA7 = np.array([], dtype=np.float32)
        self.EMG_DATA8 = np.array([], dtype=np.float32)
        self.emg_plot1 = self.graphWidget1.plot(pen=self.pen)
        self.emg_plot2 = self.graphWidget2.plot(pen=self.pen)
        self.emg_plot3 = self.graphWidget3.plot(pen=self.pen)
        self.emg_plot4 = self.graphWidget4.plot(pen=self.pen)
        self.emg_plot5 = self.graphWidget5.plot(pen=self.pen)
        self.emg_plot6 = self.graphWidget6.plot(pen=self.pen)
        self.emg_plot7 = self.graphWidget7.plot(pen=self.pen)
        self.emg_plot8 = self.graphWidget8.plot(pen=self.pen)

        self.timer1.timeout.connect(self.update_upper_graph1)
        self.timer1.start(50)

    def on_click_EMG2_Lower(self):

        if len(self.file_list2) == 1:
            filename = self.filename2
            if self.sub5.currentText() == "Overhead Clear":
                index_start = (int(self.sub6.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub6.currentText()) - 1) * 2 + self.over2 * 2
        else :
            index_start = (int(self.sub6.currentText()) - 1) * 2
            if self.sub5.currentText() == "Overhead Clear":
                filename = self.filename2
            else:
                filename = self.filename4

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            EMG = f['gforce-lowerarm-emg']
            EMG_data = EMG['emg-values']['time_str']

            data_index_start = timeSearch(EMG_data, time1)
            data_index_stop = timeSearch(EMG_data, time2)

            searched_data = EMG['emg-values']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data2 = np.array_split(searched_data, split_index)

        self.x = searched_data.shape[0]
        self.y = searched_data[:,0]

        self.clearWidget(self.vbox3)

        self.vbox3.addWidget(QWidget())

        label = QLabel('P2 Lowerarm EMG')
        label.setStyleSheet("QLabel { qproperty-alignment: AlignCenter; }")
        self.vbox3.layout().addWidget(label)


        self.graphWidget9 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget9)
        self.graphWidget9.setBackground('w')

        self.graphWidget10 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget10)
        self.graphWidget10.setBackground('w')

        self.graphWidget11 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget11)
        self.graphWidget11.setBackground('w')

        self.graphWidget12 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget12)
        self.graphWidget12.setBackground('w')

        self.graphWidget13 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget13)
        self.graphWidget13.setBackground('w')

        self.graphWidget14 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget14)
        self.graphWidget14.setBackground('w')

        self.graphWidget15 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget15)
        self.graphWidget15.setBackground('w')

        self.graphWidget16 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget16)
        self.graphWidget16.setBackground('w')

        self.EMG_DATA9 = np.array([], dtype=np.float32)
        self.EMG_DATA10 = np.array([], dtype=np.float32)
        self.EMG_DATA11 = np.array([], dtype=np.float32)
        self.EMG_DATA12 = np.array([], dtype=np.float32)
        self.EMG_DATA13 = np.array([], dtype=np.float32)
        self.EMG_DATA14 = np.array([], dtype=np.float32)
        self.EMG_DATA15 = np.array([], dtype=np.float32)
        self.EMG_DATA16 = np.array([], dtype=np.float32)
        self.emg_plot9 = self.graphWidget9.plot(pen=self.pen)
        self.emg_plot10 = self.graphWidget10.plot(pen=self.pen)
        self.emg_plot11 = self.graphWidget11.plot(pen=self.pen)
        self.emg_plot12 = self.graphWidget12.plot(pen=self.pen)
        self.emg_plot13 = self.graphWidget13.plot(pen=self.pen)
        self.emg_plot14 = self.graphWidget14.plot(pen=self.pen)
        self.emg_plot15 = self.graphWidget15.plot(pen=self.pen)
        self.emg_plot16 = self.graphWidget16.plot(pen=self.pen)

        self.timer2.timeout.connect(self.update_upper_graph2)
        self.timer2.start(50)

    def on_click_Eye(self):
        self.splited_data1 = []
        self.timer1.stop()
        self.timer1.deleteLater()
        self.timer1 = QTimer(self)
        self.update_count1 = 0
        self.splited_data2 = []
        self.timer2.stop()
        self.timer2.deleteLater()
        self.timer2 = QTimer(self)
        self.update_count2 = 0
        self.on_click_Eye1()
        self.on_click_Eye2()

    def on_click_Eye1(self):

        if len(self.file_list1) == 1:
            filename = self.filename1
            if self.sub2.currentText() == "Overhead Clear":
                index_start = (int(self.sub3.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub3.currentText()) - 1) * 2 + self.over1 * 2
        else :
            index_start = (int(self.sub3.currentText()) - 1) * 2
            if self.sub2.currentText() == "Overhead Clear":
                filename = self.filename1
            else:
                filename = self.filename3

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            EMG = f['eye-gaze']
            Eye_data = EMG['gaze']['time_str']

            data_index_start = timeSearch(Eye_data, time1)
            data_index_stop = timeSearch(Eye_data, time2)

            searched_data = EMG['gaze']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data1 = np.array_split(searched_data, split_index)

        self.clearWidget(self.vbox2)

        self.vbox2.addWidget(QWidget())

        self.graphWidget1 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget1)
        self.graphWidget1.setBackground('w')
        self.graphWidget1.setTitle("P1 Gaze X")

        self.graphWidget2 = pg.PlotWidget()
        self.vbox2.layout().addWidget(self.graphWidget2)
        self.graphWidget2.setBackground('w')
        self.graphWidget2.setTitle("P1 Gaze Y")

        self.EMG_DATA1 = np.array([], dtype=np.float32)
        self.EMG_DATA2 = np.array([], dtype=np.float32)
        self.emg_plot1 = self.graphWidget1.plot(pen=self.pen)

        self.emg_plot2 = self.graphWidget2.plot(pen=self.pen)

        self.timer1.timeout.connect(self.update_eye_graph1)
        self.timer1.start(50)

    def on_click_Eye2(self):

        if len(self.file_list2) == 1:
            filename = self.filename2
            if self.sub5.currentText() == "Overhead Clear":
                index_start = (int(self.sub6.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub6.currentText()) - 1) * 2 + self.over2 * 2
        else :
            index_start = (int(self.sub6.currentText()) - 1) * 2
            if self.sub5.currentText() == "Overhead Clear":
                filename = self.filename2
            else:
                filename = self.filename4

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            EMG = f['eye-gaze']
            Eye_data = EMG['gaze']['time_str']

            data_index_start = timeSearch(Eye_data, time1)
            data_index_stop = timeSearch(Eye_data, time2)

            searched_data = EMG['gaze']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data2 = np.array_split(searched_data, split_index)

        self.clearWidget(self.vbox3)

        self.vbox3.addWidget(QWidget())

        self.graphWidget3 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget3)
        self.graphWidget3.setBackground('w')
        self.graphWidget3.setTitle("P2 Gaze X")

        self.graphWidget4 = pg.PlotWidget()
        self.vbox3.layout().addWidget(self.graphWidget4)
        self.graphWidget4.setBackground('w')
        self.graphWidget4.setTitle("P2 Gaze Y")


        self.EMG_DATA3 = np.array([], dtype=np.float32)
        self.EMG_DATA4 = np.array([], dtype=np.float32)
        self.emg_plot3 = self.graphWidget3.plot(pen=self.pen)
        self.emg_plot4 = self.graphWidget4.plot(pen=self.pen)

        self.timer2.timeout.connect(self.update_eye_graph2)
        self.timer2.start(50)

    def on_click_Insole(self):
        self.splited_data1 = []
        self.timer1.stop()
        self.timer1.deleteLater()
        self.timer1 = QTimer(self)
        self.update_count1 = 0
        self.splited_data2 = []
        self.timer2.stop()
        self.timer2.deleteLater()
        self.timer2 = QTimer(self)
        self.update_count2 = 0
        self.on_click_Insole1()
        self.on_click_Insole2()

    def on_click_Insole1(self):

        if len(self.file_list1) == 1:
            filename = self.filename1
            if self.sub2.currentText() == "Overhead Clear":
                index_start = (int(self.sub3.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub3.currentText()) - 1) * 2 + self.over1 * 2
        else :
            index_start = (int(self.sub3.currentText()) - 1) * 2
            if self.sub2.currentText() == "Overhead Clear":
                filename = self.filename1
            else:
                filename = self.filename3

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            Insole_left = f['moticon-insole']
            Insole_right = f['moticon-insole']
            Insole_data = Insole_left['left-pressure']['time_str']

            data_index_start = timeSearch(Insole_data, time1)
            data_index_stop = timeSearch(Insole_data, time2)

            searched_data_left = Insole_left['left-pressure']['data'][data_index_start:data_index_stop]
            searched_data_right = Insole_right['right-pressure']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 40) + 1)
            self.splited_data1 = np.array_split(searched_data_left, split_index)
            self.splited_data2 = np.array_split(searched_data_right, split_index)

        self.clearWidget(self.vbox2)

        self.vbox2.addWidget(QWidget())

        self.fig1, self.ax1 = plt.subplots()
        self.fig1.suptitle('P1 Left')
        self.heatmap1 = self.ax1.imshow(self.splited_data1[0][-1,:].reshape(1,16), cmap='coolwarm', aspect= 2)
        plt.colorbar(self.heatmap1)
        self.canvas1 = FigureCanvas(self.fig1)

        self.fig2, self.ax2 = plt.subplots()
        self.fig2.suptitle('P1 Right')
        self.heatmap2 = self.ax2.imshow(self.splited_data2[0][-1,:].reshape(1,16), cmap='coolwarm', aspect= 2)
        plt.colorbar(self.heatmap2)
        self.canvas2 = FigureCanvas(self.fig2)

        self.vbox2.addWidget(self.canvas1)
        self.vbox2.addWidget(self.canvas2)

        self.timer1.timeout.connect(self.update_insole_graph1)
        self.timer1.start(25)

    def on_click_Insole2(self):

        if len(self.file_list2) == 1:
            filename = self.filename2
            if self.sub5.currentText() == "Overhead Clear":
                index_start = (int(self.sub6.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub6.currentText()) - 1) * 2 + self.over2 * 2
        else :
            index_start = (int(self.sub6.currentText()) - 1) * 2
            if self.sub5.currentText() == "Overhead Clear":
                filename = self.filename2
            else:
                filename = self.filename4

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            Insole_left = f['moticon-insole']
            Insole_right = f['moticon-insole']
            Insole_data = Insole_left['left-pressure']['time_str']

            data_index_start = timeSearch(Insole_data, time1)
            data_index_stop = timeSearch(Insole_data, time2)

            searched_data_left = Insole_left['left-pressure']['data'][data_index_start:data_index_stop]
            searched_data_right = Insole_right['right-pressure']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 40) + 1)
            self.splited_data3 = np.array_split(searched_data_left, split_index)
            self.splited_data4 = np.array_split(searched_data_right, split_index)

        self.clearWidget(self.vbox3)

        self.vbox3.addWidget(QWidget())

        self.fig3, self.ax3 = plt.subplots()
        self.fig3.suptitle('P2 Left')
        self.heatmap3 = self.ax3.imshow(self.splited_data3[0][-1,:].reshape(1,16), cmap='coolwarm', aspect= 2)
        plt.colorbar(self.heatmap3)
        self.canvas3 = FigureCanvas(self.fig3)

        self.fig4, self.ax4 = plt.subplots()
        self.fig4.suptitle('P2 Right')
        self.heatmap4 = self.ax4.imshow(self.splited_data4[0][-1,:].reshape(1,16), cmap='coolwarm', aspect= 2)
        plt.colorbar(self.heatmap4)
        self.canvas4 = FigureCanvas(self.fig4)

        self.vbox3.addWidget(self.canvas3)
        self.vbox3.addWidget(self.canvas4)

        self.timer2.timeout.connect(self.update_insole_graph2)
        self.timer2.start(25)

    def on_click_PNS(self):
        self.splited_data1 = []
        self.timer1.stop()
        self.timer1.deleteLater()
        self.timer1 = QTimer(self)
        self.update_count1 = 0
        self.splited_data2 = []
        self.timer2.stop()
        self.timer2.deleteLater()
        self.timer2 = QTimer(self)
        self.update_count2 = 0
        self.on_click_PNS1()
        self.on_click_PNS2()

    def on_click_PNS1(self):

        if len(self.file_list1) == 1:
            filename = self.filename1
            if self.sub2.currentText() == "Overhead Clear":
                index_start = (int(self.sub3.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub3.currentText()) - 1) * 2 + self.over1 * 2
        else :
            index_start = (int(self.sub3.currentText()) - 1) * 2
            if self.sub2.currentText() == "Overhead Clear":
                filename = self.filename1
            else:
                filename = self.filename3

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            PNS = f['pns-joint']
            position_data = PNS['global-position']['time_str']

            data_index_start = timeSearch(position_data, time1)
            data_index_stop = timeSearch(position_data, time2)

            searched_data = PNS['global-position']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data1 = np.array_split(searched_data, split_index)

        self.clearWidget(self.vbox2)

        self.vbox2.addWidget(QWidget())

        fig, axs = plt.subplots(nrows=1, ncols=1,
                                squeeze=False,  # if False, always return 2D array of axes
                                sharex=True, sharey=True,
                                subplot_kw={
                                    'frame_on': True,
                                    'projection': '3d',
                                },
                                figsize=(7, 5)
                                )
        ax = axs[0][0]
        fig.suptitle('P1')
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.view_init(20, -40)  # (elevation, azimuth) # (20, -40)

        self.fig1 = fig
        self.ax1 = ax

        self.x1 = searched_data[0][0]
        self.z1 = searched_data[0][2]

        self.canvas1 = FigureCanvas(self.fig1)

        self.vbox2.addWidget(self.canvas1)

        self.timer1.timeout.connect(self.update_pns_graph1)
        self.timer1.start(50)

    def on_click_PNS2(self):

        if len(self.file_list2) == 1:
            filename = self.filename2
            if self.sub5.currentText() == "Overhead Clear":
                index_start = (int(self.sub6.currentText()) - 1) * 2
            else :
                index_start = (int(self.sub6.currentText()) - 1) * 2 + self.over2 * 2
        else :
            index_start = (int(self.sub6.currentText()) - 1) * 2
            if self.sub5.currentText() == "Overhead Clear":
                filename = self.filename2
            else:
                filename = self.filename4

        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['time_str']
            time1 = data[index_start]
            time2 = data[index_start + 1]

            PNS = f['pns-joint']
            position_data = PNS['global-position']['time_str']

            data_index_start = timeSearch(position_data, time1)
            data_index_stop = timeSearch(position_data, time2)

            searched_data = PNS['global-position']['data'][data_index_start:data_index_stop]
            date_format = "%Y-%m-%d %H:%M:%S.%f"
            date1 = datetime.strptime(time2[0].decode('utf-8'), date_format)
            date2 = datetime.strptime(time1[0].decode('utf-8'), date_format)
            split_index = int(((date1 - date2).total_seconds() * 20) + 1)
            self.splited_data2 = np.array_split(searched_data, split_index)

        self.clearWidget(self.vbox3)

        self.vbox3.addWidget(QWidget())

        fig, axs = plt.subplots(nrows=1, ncols=1,
                                squeeze=False,  # if False, always return 2D array of axes
                                sharex=True, sharey=True,
                                subplot_kw={
                                    'frame_on': True,
                                    'projection': '3d',
                                },
                                figsize=(7, 5)
                                )
        ax = axs[0][0]
        fig.suptitle('P2')
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.view_init(20, -40)  # (elevation, azimuth) # (20, -40)

        self.fig2 = fig
        self.ax2 = ax

        self.x2 = searched_data[0][0]
        self.z2 = searched_data[0][2]

        self.canvas2 = FigureCanvas(self.fig2)

        self.vbox3.addWidget(self.canvas2)
        self.timer2.timeout.connect(self.update_pns_graph2)
        self.timer2.start(50)

    def clearWidget(self, w):
        try:
            for i in reversed(range(w.count())):
                w.itemAt(i).widget().deleteLater()
        except:
            print("hi2")

    def searchActivities(self, filename):
        with h5py.File(filename, "r") as f:
            dataset = f['experiment-activities']
            data = dataset['activities']['data']
            size = data.shape[0]
            half = int(size / 2)

            if data[half].tolist()[0] == b'' :
                size = 400
                half = 200

            i = 1
            if b'Backhand Driving' in data[half].tolist()[0]:
                while True:
                    try:
                        if b'Backhand Driving' not in data[half - i].tolist()[0]:
                            if b'Overhead Clear' in data[half - i].tolist()[0]:
                                over = (half - i + 1) / 2
                            else :
                                over = 0
                            back = half - over
                            break
                        else:
                            i = i + 1
                    except:
                        over = 0
                        back = half
                        break
            else :
                while True:
                    try:
                        if b'Overhead Clear' not in data[half + i].tolist()[0]:
                            over = (half + i) / 2
                            if b'Backhand Driving' in data[half + i].tolist()[0]:
                                back = half - over
                            else :
                                back = 0
                            break
                        else:
                            i = i + 1
                    except:
                        over = (half + i) / 2
                        back = 0
                        break

            return int(over), int(back)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Visualizer()
    sys.exit(app.exec_())
