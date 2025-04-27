import torch
import cv2
from ultralytics import YOLO
from config import MODELV8N_PATH

# 加载模型
model = YOLO(MODELV8N_PATH)

def simple_preprocess(image):
    # resize到640×640，并归一化
    image = cv2.resize(image, (640, 640))  # Resize to model's expected input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H,W,C) -> (C,H,W)
    image = image / 255.0  # normalize to 0-1
    return image

def get_raw_output(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    im_tensor = simple_preprocess(image)  # 手动处理成tensor

    im_tensor = im_tensor.unsqueeze(0)  # 加上batch维度 [1, 3, 640, 640]
    im_tensor.requires_grad_(True)  # 开启梯度

    with torch.set_grad_enabled(True):
        preds = model.model(im_tensor)  # 手动forward

    return preds[0], im_tensor
