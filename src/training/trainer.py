import os
import shutil
import subprocess
import zipfile
import torch

class YOLOv5Trainer:
    def __init__(self, yolov5_path="yolov5"):
        self.yolov5_path = yolov5_path

    def extract_zip(self, zip_path, extract_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    def train(self, data_path, batch_size, epochs, output_callback=None):
        # Ensure yolov5 directory exists
        if not os.path.exists(self.yolov5_path):
            if output_callback:
                output_callback("Error: YOLOv5 directory not found. Please clone it into the project root.")
            return False

        current_dir = os.getcwd()
        os.chdir(self.yolov5_path)

        command = f"python train.py --img 416 --batch {batch_size} --epochs {epochs} --data {data_path}/data.yaml --weights yolov5s.pt --cache"
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        if output_callback:
            for line in process.stdout:
                output_callback(line.strip())

        os.chdir(current_dir)
        return True

    def save_model(self, destination_dir="models"):
        weights_file = os.path.join(self.yolov5_path, "runs/train/exp/weights/best.pt")
        
        if not os.path.exists(weights_file):
            print(f"Error: Model weights not found at {weights_file}")
            return False

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        destination_path = os.path.join(destination_dir, "best.pt")
        shutil.copy2(weights_file, destination_path)
        return True

    def get_device_info(self):
        return f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})"
