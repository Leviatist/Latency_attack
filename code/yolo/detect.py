from ultralytics import YOLO

from config import MODEL_PATH

model = YOLO(MODEL_PATH)

def detect_image(image):
    results = model(image, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2, conf = box.xyxy[0].tolist() + [box.conf[0].item()]
            detections.append([x1, y1, x2, y2, conf])
    return detections
