import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
import os
def main():
    #model_path="outputs/train/test3/checkpoints/last"
    #model_path=os.path.join( "outputs/train/test3/checkpoints/",  os.readlink(model_path))
    #step,policy, optimizer, lr_scheduler=load_checkpoint(checkout_dir=model_path)
    #step,policy, optimizer, lr_scheduler=load_pretrained_model(model_path=model_path, dataset_path="outputs/dataset/test3")
    #step,policy, optimizer, lr_scheduler=load_pretrained_model(model_path="D:/github/lerobot/outputs/train/test3", dataset_path="D:/github/lerobot/outputs/dataset/test3")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
