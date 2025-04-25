import cv2
import torch
import numpy as np
from ultralytics import YOLO
from config import MODELV8N_PATH, IMAGE_PATH
import matplotlib.pyplot as plt

# 1. 加载模型
model = YOLO(MODELV8N_PATH)  # 加载自定义路径的 YOLO 模型

# 2. 加载并处理输入图像
img = cv2.imread(IMAGE_PATH)  # 读取图像

# 将 BGR 转换为 RGB 格式
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3. 调整图像大小为 640x640，确保符合 YOLO 模型要求
img_resized = cv2.resize(img_rgb, (640, 640))

# 转换为 PyTorch 张量，并进行标准化
img_tensor = torch.from_numpy(img_resized).float() / 255.0  # 标准化到 [0, 1]
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # 转换为 [1, 3, H, W] 格式

# 4. 设置 requires_grad=True，以便计算梯度
img_tensor.requires_grad_()

# 5. 模型推理并计算损失
# 通过模型进行预测
output = model(img_tensor)
# 获取预测框和标签
boxes = output.boxes  # 预测框
confidences = boxes.conf  # 置信度
labels = boxes.cls  # 类别标签

# 选择一个目标类别进行梯度计算，假设我们选择 "bird"（类别ID 14）
target_class_id = 14
target_mask = labels == target_class_id  # 选择目标类别的检测结果

# 如果有目标类别的检测框
if target_mask.any():
    # 选择第一个符合目标类别的框
    target_box = boxes[target_mask][0]

    # 自定义损失函数：基于预测框的置信度来计算损失
    loss = -confidences[target_mask][0]  # 最大化置信度

    # 6. 计算损失对输入图像的梯度
    loss.backward()  # 反向传播计算梯度

    # 获取输入图像的梯度
    gradients = img_tensor.grad