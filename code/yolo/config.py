import os

abs_path = os.path.abspath(__file__)
pro_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))

MODEL11X_PATH = os.path.join(pro_path, 'model/yolo11x.pt')
MODELV8N_PATH = os.path.join(pro_path, 'model/yolov8n.pt')
IMAGE_PATH = os.path.join(pro_path, 'data/img/0001.jpg')
OUTPUT_PATH = os.path.join(pro_path, 'output/attacked.jpg')
CSV_PATH = os.path.join(pro_path, 'data/csv/preds.csv')

