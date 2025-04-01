import multiprocessing.process
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit,
    QListWidget, QMessageBox,QPushButton,
    QTreeWidget, QTreeWidgetItem, QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import pyqtSignal
import os
import torch
from lerobot.scripts.train import train_with_config
from lerobot.scripts.train_plus import load_pretrained_model 
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
import multiprocessing
from lerobot.common.policies.act.configuration_act import ACTConfig

from lerobot.common.utils.utils import init_logging

# Set models directory
models_dir = "outputs/train" #dont change 
dataset_dir = "outputs/dataset" #dont change 

current_policy:PreTrainedPolicy=None
current_data_name:str=None
current_model_path:str=None

class ModelPanel(QWidget):
    model_loaded = pyqtSignal(str)  # Signal emitted when model is loaded
    #def __reduce__(self):
    #    return (self.__class__,(self.arg1, self.arg2))
    
    def __init__(self):
        super().__init__()
        
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel for dataset tree
        self.dataset_tree = QTreeWidget()
        self.dataset_tree.setHeaderLabel("Datasets")
        main_layout.addWidget(self.dataset_tree, stretch=1)
        
        # middle panel for model dir
        middle_panel = QWidget()
        layout = QVBoxLayout()
        middle_panel.setLayout(layout)
        main_layout.addWidget(middle_panel, stretch=1)

        # Create model list
        self.model_list = QListWidget()
        layout.addWidget(self.model_list)


        # Model name
        mn_layout = QHBoxLayout()
        mn_layout.addWidget(QLabel("Model name:"))
        self.mn_input = QTextEdit()
        self.mn_input.setPlaceholderText("Enter model name") 
        #self.left_com_input.setText("")
        self.mn_input.setMaximumHeight(30)
        mn_layout.addWidget(self.mn_input)
        layout.addLayout(mn_layout)

        # Policy type
        p_layout = QHBoxLayout()
        p_layout.addWidget(QLabel("Policy type:"))
        self.pt_input = QTextEdit()
        self.pt_input.setPlaceholderText("Enter policy type (eg. act)") 
        self.pt_input.setText("act")
        self.pt_input.setMaximumHeight(30)
        p_layout.addWidget(self.pt_input)
        layout.addLayout(p_layout)

        # save frequence
        sf_layout = QHBoxLayout()
        sf_layout.addWidget(QLabel("Save frequence:"))
        self.sf_input = QTextEdit()
        self.sf_input.setPlaceholderText("Enter save frequence (eg. 20000)") 
        self.sf_input.setText("2000")
        self.sf_input.setMaximumHeight(30)
        sf_layout.addWidget(self.sf_input)
        layout.addLayout(sf_layout)


        self.add_button=QPushButton("Creat Model")
        layout.addWidget(self.add_button)
        # Create load button
        self.load_button = QPushButton("Load Model")
        layout.addWidget(self.load_button)
    
        # Buttons
        self.save_button = QPushButton("Save Model")
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        
        self.train_button = QPushButton("Train Model")
        self.train_button.setEnabled(False)
        layout.addWidget(self.train_button)
        
        right_panel= QWidget()
        model_info= QVBoxLayout()
        right_panel.setLayout(model_info)
        # Model info display
        self.model_info_label = QLabel("No model loaded")
        
        model_info.addWidget(self.model_info_label)
        
        self.model_details = QTextEdit()
        self.model_details.setReadOnly(True)
        model_info.addWidget(self.model_details)
        main_layout.addWidget(right_panel,stretch=1)


        self.refresh_datasets_dir()
        self.refresh_model_dir()
      
    def setup_connections(self):
        self.dataset_tree.itemDoubleClicked.connect(self.on_dataset_double_click)
        self.add_button.clicked.connect(self.handle_model_create)
        self.save_button.clicked.connect(self.save_model)
        self.train_button.clicked.connect(self.train_model)
        
        # Connect signals
        self.load_button.clicked.connect( self.handle_model_load)
        self.train_button.clicked.connect( self.handle_model_train )
        
        
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
            
        current_data_name = item.text(0)
        current_data_path = os.path.join(dataset_dir, current_data_name)
         
        if os.path.exists(current_data_path):
            print("current path:", current_data_path)
        else:
            print("wrong path")
        
        #    item.takeChildren()  # Clear any existing subitems
        #    for data_item in os.listdir(current_data_path):
        #        QTreeWidgetItem(item, [data_item])
    
    def create_model(self):
        train_with_config(repo_id=current_data_name, output_dir=current_model_path+self.mn_input.toPlainText(),
                          pretrained_path=None, train_steps=100000, save_freq=int(self.sf_input.toPlainText()))
        self.refresh_model_dir()

    def handle_model_create(self):
        self.process=multiprocessing.Process(target=self.create_model)
        self.process.start()

        
    
    def handle_model_load(self):
        """Handle model loading from model selector"""
        model_path = self.get_selected_model_path()
        if model_path:
            self.load_model(model_path)
            print("current model path : ", model_path)
    
    
    def handle_model_train(self):
        """Handle model training"""
        self.model_panel.train_model()
    
    def on_dataset_double_click_model(self, item, column):
        """Handle double click on dataset item"""
        if item.text(0) == "No datasets found":
            return
            
        model_name = item.text(0)
        current_model_path = os.path.join(models_dir, model_name)
         
        if os.path.exists(current_model_path):
            print("current path:", current_model_path)
        else:
            print("wrong path")
        
        #    item.takeChildren()  # Clear any existing subitems
        #    for data_item in os.listdir(self.current_data_path):
        #        QTreeWidgetItem(item, [data_item])
    def refresh_model_dir(self):
        """Refresh model list"""
        self.model_list.clear()
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path):
                self.model_list.addItem(model_name)
                
    def get_selected_model_path(self):
        """Get selected model path"""
        selected = self.model_list.currentItem()
        if selected:
            model_name = selected.text()
            return os.path.join(models_dir, model_name,"checkpoints")
        return None



    def load_model(self, model_path):
        """Load and display model details"""
        try:
            
            aux=os.readlink(model_path+"/last")
            current_model_path=os.path.join( model_path,  aux)
            current_data_path= os.path.join(dataset_dir,current_data_name)
            step,current_policy,optimizer, lr_scheduler=load_pretrained_model(model_path=current_model_path, dataset_path=current_data_path)
            
            
            
            self.model_info_label.setText(f"Loaded model: {model_path}")
            
            # Display model details
            model_info = f"Model architecture:\n{str(current_policy)}\n\n"
            model_info += f"Model parameters: {sum(p.numel() for p in current_policy.parameters())}"
            self.model_details.setText(model_info)
            
            # Enable buttons
            self.save_button.setEnabled(True)
            self.train_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load model: {str(e)}"
            )
    
    def save_model(self, save_path=None):
        """Save model to new location"""
        if not current_policy:
            return
            
        try:
            if not save_path:
                options = QFileDialog.Options()
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Model", "", 
                    "Model Files (*.pt *.pth *.ckpt);;All Files (*)",
                    options=options
                )
                
            if save_path:
                torch.save(current_policy, save_path)
                QMessageBox.information(
                    self, "Success", 
                    f"Model saved to {save_path}"
                )
                return True
                
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to save model: {str(e)}"
            )
        return False
    
    def train_model(self):
        """Train the current model"""
        if not current_policy:
            return
            
        try:
            
            # TODO: Implement actual training logic
            # For now just show a message
            QMessageBox.information(
                self, "Training Started",
                "Model training has begun..."
            )
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to train model: {str(e)}"
            )
            return False


if __name__ == "__main__":
    init_logging()
    #train_with_config(repo_id='fr3/test_wzl', output_dir=models_dir+'fr3/test_wzl', pretrained_path='outputs/train/fr2/checkpoints/last/', train_steps=131411)
    train_with_config(repo_id='fr3/test_wzl', output_dir=models_dir+'fr3/test_wzl', pretrained_path=None, train_steps=131411)         