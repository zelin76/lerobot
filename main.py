import sys
from PyQt6.QtWidgets import QApplication

from lerobot.scripts.train_plus import load_pretrained_model
import os
def main():
    model_path="outputs/train/test3/checkpoints/last"
    model_path=os.path.join( "outputs/train/test3/checkpoints/",  os.readlink(model_path))
    step,policy, optimizer, lr_scheduler=load_pretrained_model(model_path=model_path, dataset_path="outputs/dataset/test3")
    print("every thing ok")

if __name__ == "__main__":
    main()
