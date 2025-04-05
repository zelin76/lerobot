from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit,
    QListWidget, QMessageBox,QPushButton,
    QTreeWidget, QTreeWidgetItem, QHBoxLayout, QGroupBox,QRadioButton,QApplication
)
import cv2
from PyQt6.QtCore import pyqtSignal, QTimer,Qt
import os
import torch
from PyQt6.QtGui import QImage, QPixmap
from lerobot.common.robot_devices.control_configs import (
    ControlConfig,
    ControlPipelineConfig
    )
from lerobot.scripts.control_robot_server import control_robot_server,ServerControlConfig, shared_dict,queue
from lerobot.scripts.control_robot_client import control_robot_client,ClientControlConfig, client_level
import numpy as np
from lerobot.scripts.baisic import *
import multiprocessing
from multiprocessing import Queue,shared_memory
# Set models directory
from lerobot.common.utils.utils import init_logging
 

totoImage=np.zeros((240, 960, 3), dtype=np.uint8)
def parent_consumer(queue):
    """父进程：从共享内存读取图像并清理"""
    # 从Queue获取共享内存信息
    data = queue.get()
    if data is not None:
        
        shm_name, shape, dtype = data["name"], data["shape"], data["dtype"]
    
        # 连接到共享内存并读取数据
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        image = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf).copy()  # 复制数据避免后续unlink影响
    
        # 销毁共享内存
        #existing_shm.close()
        #existing_shm.unlink()
        #print("[父进程] 图像已显示，共享内存已销毁")
        return existing_shm,image
    else :
        return None , None

def consumer(shared_dict):
    # 从共享字典中读取数据并重建图像
    img_bytes = shared_dict.get("img_data")
    if img_bytes:
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 解码为图像
        return img
    else:
        return None


class DataPanel(QWidget):
    model_loaded = pyqtSignal(str)  # Signal emitted when model is loaded


    def __init__(self):
        super().__init__()
        
        self.current_data_path=None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        self.robot_thread = None
        self.timer=QTimer(self)
        self.timer.setInterval(33)
        self.timer.start()

        main0_layout = QVBoxLayout()
        
        
        
        self.setLayout(main0_layout)
        
        #display images
        self.image_layout=ndarray_to_qimage(totoImage)
        self.qpixmap=QPixmap.fromImage(self.image_layout)
        self.image_label=QLabel()
        self.image_label.setPixmap(self.qpixmap)
        main0_layout.addWidget(self.image_label) 
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        other_data=QWidget()
        main0_layout.addWidget(other_data)
        
        
        main_layout=QHBoxLayout()
        other_data.setLayout(main_layout) 
        # Left panel for dataset tree
        self.dataset_tree = QTreeWidget()
        self.dataset_tree.setHeaderLabel("Datasets")
        main_layout.addWidget(self.dataset_tree, stretch=1)
        
        # middle panel for model dir
        middle_panel = QWidget()
        
       
        type_group = QGroupBox("Record Type")
        type_layout = QVBoxLayout()
        

        self.server_radio = QRadioButton("Server Record")
        self.client_radio = QRadioButton("Client Record") 
        self.client_radio.setChecked(True)  # Default selection
        
       
        type_layout.addWidget(self.server_radio)
        type_layout.addWidget(self.client_radio)
        
        
        
        type_group.setLayout(type_layout)
        
         # Add dataset button
        self.add_btn = QPushButton("Add dataset")
        type_layout.addWidget(self.add_btn)
        
        
        middle_panel.setLayout(type_layout)
        main_layout.addWidget(middle_panel, stretch=1)
        
        # Right panel for dataset creation
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
         
        # Dataset ip
        ip_layout = QHBoxLayout()
        right_layout.addLayout(ip_layout)
        ip_layout.addWidget(QLabel("Server ip:"))
        self.data_ip_input = QTextEdit()
        self.data_ip_input.setPlaceholderText("Enter server ip")
        self.data_ip_input.setText("127.0.0.1")
        self.data_ip_input.setMaximumHeight(30)
        ip_layout.addWidget(self.data_ip_input)
    
        # Dataset name input
        name_layout = QHBoxLayout()
        right_layout.addLayout(name_layout)
        name_layout.addWidget(QLabel("Dataset Name:"))
        self.data_name_input = QTextEdit()
        self.data_name_input.setPlaceholderText("Enter dataset name")
        self.data_name_input.setText("test5")
        self.data_name_input.setMaximumHeight(30)
        name_layout.addWidget(self.data_name_input)
        
        
        # Data count input
        num_layout = QHBoxLayout()
        right_layout.addLayout(num_layout)
        num_layout.addWidget(QLabel("Data Count:"))
        self.data_count_input = QTextEdit()
        self.data_count_input.setPlaceholderText("Enter number of data points") 
        self.data_count_input.setText("1")
        self.data_count_input.setMaximumHeight(30)
        num_layout.addWidget(self.data_count_input)
        
        
        # left com input
        lcom_layout = QHBoxLayout()
        lcom_layout.addWidget(QLabel("Left serial:"))
        self.left_com_input = QTextEdit()
        self.left_com_input.setPlaceholderText("Enter serial port name") 
        self.left_com_input.setText("/dev/ttyACM4")
        self.left_com_input.setMaximumHeight(30)
        lcom_layout.addWidget(self.left_com_input)
        right_layout.addLayout(lcom_layout)
        
        # right com input
        rcom_layout = QHBoxLayout()
        rcom_layout.addWidget(QLabel("Right serial:"))
        self.right_com_input = QTextEdit()
        self.right_com_input.setPlaceholderText("Enter serial port name") 
        self.right_com_input.setText("/dev/ttyACM3")
        self.right_com_input.setMaximumHeight(30)
        rcom_layout.addWidget(self.right_com_input)
        right_layout.addLayout(rcom_layout)
        
        main_layout.addWidget(right_panel, stretch=1)
       

        self.refresh_datasets_dir()
        
      
    def setup_connections(self):
        self.dataset_tree.itemDoubleClicked.connect(self.on_dataset_double_click)
        self.add_btn.clicked.connect(self.handle_add_dataset)
        self.timer.timeout.connect(self._update_camera_views)
     
    def refresh_datasets_dir(self):
        """Refresh the dataset tree from outputs/dataset directory"""
        self.dataset_tree.clear()
        
       
        if os.path.exists(dataset_dir):
            for item1 in os.listdir(dataset_dir): # data names
                level1_item=QTreeWidgetItem(self.dataset_tree)
                level1_item.setText(0,item1)
                # next level2
                sub1_dir=os.path.join(dataset_dir, item1)
                if os.path.isdir(sub1_dir):
                    for item2 in os.listdir(sub1_dir):
                        level2_item=QTreeWidgetItem(level1_item)
                        level2_item.setText(0,item2)
                        
                        sub2_dir= os.path.join(sub1_dir, item2)
                        if os.path.isdir(sub2_dir):
                            for item3 in os.listdir(sub2_dir):
                                level3_item=QTreeWidgetItem(level2_item)
                                level3_item.setText(0,item3)
                                sub3_dir= os.path.join(sub2_dir, item3)
                                if os.path.isdir(sub3_dir):
                                    for item4 in os.listdir(sub3_dir):
                                        level4_item=QTreeWidgetItem(level3_item)
                                        level4_item.setText(0,item4)
        else:
            QTreeWidgetItem(self.dataset_tree, ["No datasets found"])
            
    def on_dataset_double_click(self, item, column):
        """Handle double click on dataset item"""
        if item.text(0) == "No datasets found":
            return
            
        dataset_name = item.text(0)
        self.current_data_path = os.path.join(dataset_dir, dataset_name)
         
        if os.path.exists(self.current_data_path):
            print("current path:", self.current_data_path)
        else:
            print("wrong path")

    def _update_camera_views(self):
        global queue
        #cv2.imwrite("/home/liu/test.png",totolImage)

        
        
        if self.server_radio.isChecked() :
            if shared_dict['server_level'].value==0 or shared_dict['server_level'].value==3:
                self.add_btn.setEnabled(True)
            else : 
                self.add_btn.setEnabled(False)
                existing_shm, image=parent_consumer(queue)
                #image =consumer(shared_dict=shared_dict)
                if image is not None:
                    self.image_layout=ndarray_to_qimage(image)
                    self.qpixmap=QPixmap.fromImage(self.image_layout)
                    self.image_label.setPixmap(self.qpixmap)
        
        if self.client_radio.isChecked():
            if client_level==0 or client_level==3:
                self.add_btn.setEnabled(True)    
            #else :
                #self.add_btn.setEnabled(False)
        
    def handle_control_robot(self):
        if self.server_radio.isChecked():
            cfg=ControlPipelineConfig(control=ServerControlConfig())
            control_robot_server(cfg)
        else:
           
            control=ClientControlConfig(listen_ip=self.data_ip_input.toPlainText(), 
                                        fps=30,
                                        repo_id=self.data_name_input.toPlainText(), 
                                        num_episodes=int(self.data_count_input.toPlainText()),
                                        single_task="test"
                                        )
            control.left_com=self.left_com_input.toPlainText()
            control.right_com=self.right_com_input.toPlainText()
            
            cfg=ControlPipelineConfig(control=control)
            
            try:
                control_robot_client(cfg)
            except Exception as e:
                    print(e)
                    
            
        self.add_btn.setVisible(True)
        self.add_btn.setEnabled(True)

    def handle_add_dataset(self):
        
        self.add_btn.setEnabled(False)
        processes=[]
        self.process = multiprocessing.Process(target=self.handle_control_robot)
        self.process.start()

        #self.robot_thread  = threading.Thread(target=self.handle_control_robot)
        #self.robot_thread.start()
        
def main():
        control=ClientControlConfig(listen_ip="10.0.0.15", 
                                        single_task="test",
                                        fps=30,
                                        repo_id="haha", 
                                        num_episodes=1
                                        
                                        )
        cfg=ControlPipelineConfig(control=control)
        control_robot_client(cfg)
       
if __name__ == "__main__":
    #main()
    init_logging()
    app = QApplication([])
    ui = DataPanel()
    ui.show()
    app.exec()    