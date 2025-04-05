from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,QTextEdit,QListWidget,QApplication
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2
import os
import numpy as np
from lerobot.scripts.infer import FairinoRobotInfer,setStopValue, task_image


from model_panel import current_policy,models_dir

from lerobot.common.utils.utils import init_logging
import multiprocessing
from lerobot.scripts.baisic import *

class TaskPanel(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setup_ui()
        self._connect_signals()
        
        # 初始化摄像头
        #self.cameras = [cv2.VideoCapture(i) for i in range(3)]
        #self.timer = QTimer()
        #self.timer.timeout.connect(self._update_camera_views)
        
    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        
        self.timer=QTimer(self)
        self.timer.setInterval(33)
        self.timer.start()
        
        # 摄像头视图
       
        self.image_layout=ndarray_to_qimage(task_image)
        self.qpixmap=QPixmap.fromImage(self.image_layout)
        self.image_label=QLabel()
        self.image_label.setPixmap(self.qpixmap)
        self.layout.addWidget(self.image_label) 

        # 控制按钮
        control_layout = QHBoxLayout()
        
        # Create model list
        self.model_list = QListWidget()
        control_layout.addWidget(self.model_list)
        self.refresh_model_dir()
        
        self.run_button = QPushButton("Start")
        #self.run_button.setStyleSheet("min-width: 40px; min-height: 40px; border-radius: 20px;")
        self.pause_button = QPushButton("Pause")
        #self.pause_button.setStyleSheet("min-width: 40px; min-height: 40px; border-radius: 20px;")
        self.stop_button = QPushButton("Stop") 
        #self.stop_button.setStyleSheet("min-width: 40px; min-height: 40px; border-radius: 20px;")
        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        self.layout.addLayout(control_layout)
        
        infor_layout= QHBoxLayout()
        
        # 统计信息
        status_view=QWidget()

        stats_layout = QHBoxLayout()
        status_view.setLayout(stats_layout)
        self.task_count_label = QLabel("Number of task: 0")
        self.success_rate_label = QLabel("Suc rate: 0%")
        self.avg_time_label = QLabel("Avg time: 0s")
        self.remaining_label = QLabel("Number left: 0")
        stats_layout.addWidget(self.task_count_label)
        stats_layout.addWidget(self.success_rate_label)
        stats_layout.addWidget(self.avg_time_label)
        stats_layout.addWidget(self.remaining_label)
        infor_layout.addWidget(status_view)
        
        self.layout.addLayout(infor_layout)
        
                # Data count input
        num_layout = QHBoxLayout()
        stats_layout.addWidget(QLabel("Operation Count:"))
        stats_layout.addLayout(num_layout)
        self.op_count_input = QTextEdit()
        self.op_count_input.setPlaceholderText("Enter number of operations") 
        self.op_count_input.setText("1")
        self.op_count_input.setMaximumHeight(30)
        num_layout.addWidget(self.op_count_input)
        
        
    def _connect_signals(self):
        self.timer.timeout.connect(self._update_camera_views)
        self.run_button.clicked.connect(self._on_run)
        self.pause_button.clicked.connect(self._on_pause)
        self.stop_button.clicked.connect(self._on_stop)
        
    def _update_camera_views(self):
        # self.image_layout=ndarray_to_qimage(task_image)
        # self.qpixmap=QPixmap.fromImage(self.image_layout)
        # self.image_label.setPixmap(self.qpixmap)

        self.qpixmap=QPixmap.fromImage(self.image_layout)
        self.image_label.setPixmap(self.qpixmap)
        self.image_layout=ndarray_to_qimage(task_image)
        
    
    def turn_on_robot(self):
        symlinkpath=models_dir+self.model_list.currentItem().text()+"/checkpoints/last"
        checkpoint_path=model_path=os.path.join( models_dir+self.model_list.currentItem().text()+"/checkpoints/",  os.readlink(symlinkpath))+"/pretrained_model"
        FairinoRobotInfer(checkpoint_path,loop_time=int(self.op_count_input.toPlainText()),policy=current_policy)    
              
    def _on_run(self):
        setStopValue(0)
        self.process=multiprocessing.Process(target=self.turn_on_robot)
        self.process.start()
        
        self.run_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        
        #self.process.start()
        
    def _on_pause(self):
        setStopValue(2)

        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        
    def _on_stop(self):
        setStopValue(3)
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
    def update_stats(self, count, success_rate, avg_time, remaining):
        self.task_count_label.setText(f"任务执行次数: {count}")
        self.success_rate_label.setText(f"成功率: {success_rate}%")
        self.avg_time_label.setText(f"平均任务时长: {avg_time}s")
        self.remaining_label.setText(f"剩余任务: {remaining}")

    def refresh_model_dir(self):
        """Refresh model list"""
        self.model_list.clear()
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path):
                self.model_list.addItem(model_name)



           
                
if __name__ == "__main__":
    init_logging()
    app = QApplication([])
    ui = TaskPanel()
    ui.show()
    app.exec()