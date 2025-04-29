from ultralytics import YOLO

from lib.config import MODELV8N_PATH

model = YOLO(MODELV8N_PATH)

def detect_image(image,conf=0.3):
    results = model(image, verbose=False, conf=conf)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2, conf = box.xyxy[0].tolist() + [box.conf[0].item()]
            detections.append([x1, y1, x2, y2, conf])
    return detections
