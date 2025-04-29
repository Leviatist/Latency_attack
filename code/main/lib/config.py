import os

abs_path = os.path.abspath(__file__)
pro_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(abs_path))))

MODEL11X_PATH = os.path.join(pro_path, 'model/yolo11x.pt')
MODELV8N_PATH = os.path.join(pro_path, 'model/yolov8n.pt')
ORIGINIMG_PATH = os.path.join(pro_path, 'data/img/origin.jpg')
DUMMYIMG_PATH = os.path.join(pro_path, 'data/img/dummy.jpg')
ATKEDIMG_PATH = os.path.join(pro_path, 'data/img/atked.jpg')
PREDICTEDIMG_PATH = os.path.join(pro_path, 'data/img/predicted.jpg')
CSV_PATH = os.path.join(pro_path, 'data/csv/preds.csv')
