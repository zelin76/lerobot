from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit,
    QListWidget, QMessageBox,QPushButton,
    QTreeWidget, QTreeWidgetItem, QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import pyqtSignal, Qt, QDir
import os
import torch

from lerobot.scripts.train_plus import load_pretrained_model 
from lerobot.common.policies.pretrained import PreTrainedPolicy



# Set models directory
models_dir = "outputs/train"
dataset_dir = "outputs/dataset"
current_policy:PreTrainedPolicy
class ModelPanel(QWidget):
    model_loaded = pyqtSignal(str)  # Signal emitted when model is loaded
    
    def __init__(self):
        super().__init__()
        
        self.current_data_path=None
        
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
            
        dataset_name = item.text(0)
        self.current_data_path = os.path.join(dataset_dir, dataset_name)
         
        if os.path.exists(self.current_data_path):
            print("current path:", self.current_data_path)
        else:
            print("wrong path")
        
        #    item.takeChildren()  # Clear any existing subitems
        #    for data_item in os.listdir(self.current_data_path):
        #        QTreeWidgetItem(item, [data_item])

    
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
            
        dataset_name = item.text(0)
        self.current_data_path = os.path.join(models_dir, dataset_name)
         
        if os.path.exists(self.current_data_path):
            print("current path:", self.current_data_path)
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
            self.current_model_path=os.path.join( model_path,  aux)

            step,current_policy,optimizer, lr_scheduler=load_pretrained_model(model_path=self.current_model_path, dataset_path=self.current_data_path)
            
            #step,policy, optimizer, lr_scheduler=load_checkpoint(model_path)
            #checkpoint_dir =model_path+"/checkpoints/002000"
            
            #policy=PreTrainedPolicy
            #cfg=TrainPipelineConfig
            #optimizer=Optimizer
            #optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
            #scheduler= LRScheduler
            #step,policy, optimizer,scheduler= load_checkpoint(checkpoint_dir)
            
            
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