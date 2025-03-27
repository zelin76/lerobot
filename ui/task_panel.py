from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2

class TaskPanel(QWidget):
    def __init__(self):
        super().__init__()
        
        self._init_ui()
        self._connect_signals()
        
        # 初始化摄像头
        self.cameras = [cv2.VideoCapture(i) for i in range(3)]
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_camera_views)
        
    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        self.run_button = QPushButton("运行")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        self.layout.addLayout(control_layout)
        
        # 摄像头视图
        self.camera_layout = QHBoxLayout()
        self.camera_views = [QLabel() for _ in range(3)]
        for view in self.camera_views:
            self.camera_layout.addWidget(view)
        self.layout.addLayout(self.camera_layout)
        
        # 统计信息
        stats_layout = QVBoxLayout()
        self.task_count_label = QLabel("任务执行次数: 0")
        self.success_rate_label = QLabel("成功率: 0%")
        self.avg_time_label = QLabel("平均任务时长: 0s")
        self.remaining_label = QLabel("剩余任务: 0")
        stats_layout.addWidget(self.task_count_label)
        stats_layout.addWidget(self.success_rate_label)
        stats_layout.addWidget(self.avg_time_label)
        stats_layout.addWidget(self.remaining_label)
        self.layout.addLayout(stats_layout)
        
    def _connect_signals(self):
        self.run_button.clicked.connect(self._on_run)
        self.pause_button.clicked.connect(self._on_pause)
        self.stop_button.clicked.connect(self._on_stop)
        
    def _update_camera_views(self):
        for i, cam in enumerate(self.cameras):
            ret, frame = cam.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.camera_views[i].setPixmap(QPixmap.fromImage(q_img))
                
    def _on_run(self):
        self.timer.start(30)  # 30ms更新一次画面
        self.run_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        
    def _on_pause(self):
        self.timer.stop()
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        
    def _on_stop(self):
        self.timer.stop()
        for cam in self.cameras:
            cam.release()
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
    def update_stats(self, count, success_rate, avg_time, remaining):
        self.task_count_label.setText(f"任务执行次数: {count}")
        self.success_rate_label.setText(f"成功率: {success_rate}%")
        self.avg_time_label.setText(f"平均任务时长: {avg_time}s")
        self.remaining_label.setText(f"剩余任务: {remaining}")
