from ultralytics import YOLO
import cv2
from config import MODELV8N_PATH

# 加载模型
model = YOLO(MODELV8N_PATH)

def get_boxes_info(image_path, conf_threshold=0):
    image = cv2.imread(image_path)
    results = model(image, verbose=False, conf=0.0001)[0]

    boxes_info = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = results.names[cls]

        if conf >= 0:
            boxes_info.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "class_id": cls,
                "label": label
            })

    return boxes_info
