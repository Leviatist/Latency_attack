import cv2 
import torch
from ultralytics import YOLO

from config import MODEL_PATH,IMAGE_PATH

model = YOLO(MODEL_PATH)
image_0 = cv2.imread(IMAGE_PATH)

results = model.predict(image_0)
print(results)
print(torch.cuda.is_available())