from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter
)
from ui.model_panel import ModelPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Lerobot Model Manager")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout()
        main_widget.setLayout(layout)
        
        # Create splitter for left and right panels
        splitter = QSplitter()


        # Right side - Model Panel
        self.model_panel = ModelPanel()
        splitter.addWidget(self.model_panel)
                
        # Left side - Model Selector
        #self.model_selector = ModelSelector()
        #splitter.addWidget(self.model_selector)
         
        # Set splitter sizes
        splitter.setSizes([400, 800])
        
        layout.addWidget(splitter)
