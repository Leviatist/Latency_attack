import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from config import MODEL_PATH, IMAGE_PATH, OUTPUT_PATH, CONF_THRESHOLD, GRID_SIZE, LAMBDA, ATTACK_ITER, EPSILON
from utils import split_image, compute_l2_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(MODEL_PATH).to(device)

# 预处理图片
def preprocess_image(image, img_size=640):
    h, w, _ = image.shape
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    resized_image = cv2.resize(image, (new_w, new_h))

    # 转为torch tensor并归一化
    image_tensor = torch.from_numpy(resized_image).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return image_tensor / 255.0, resized_image

# 攻击代码
def latency_attack_pgd():
    image = cv2.imread(IMAGE_PATH)
    orig_image = image.copy()

    # 获取图片的网格分块
    grids = split_image(image, GRID_SIZE)

    # 只攻击第一个分块
    x1, y1, x2, y2, cell = grids[0]  # 只取第一块
    print(f"Attacking grid 1/1")

    perturbation = np.zeros_like(cell, dtype=np.float32)

    for iter in range(ATTACK_ITER):
        attacked_image = image.copy()
        attacked_image[y1:y2, x1:x2] = np.clip(cell + perturbation, 0, 255).astype(np.uint8)

        # 预处理图片并转换为tensor
        image_tensor, resized_image = preprocess_image(attacked_image)

        # 确保 image_tensor 可以计算梯度
        image_tensor.requires_grad = True

        # 获取模型预测
        results = model(image_tensor, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        max_conf = 0
        for det in detections:
            x1_, y1_, x2_, y2_, conf, cls = det[:6]
            cx, cy = (x1_ + x2_) / 2, (y1_ + y2_) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                max_conf = max(max_conf, conf)

        print(f"Iteration {iter+1}/{ATTACK_ITER}, Max Conf: {max_conf:.3f}")

        if max_conf > CONF_THRESHOLD:
            print(f"Grid 1: conf {max_conf:.3f} reached at iter {iter+1}")
            break

        # 真实梯度：通过目标检测回传梯度
        model.zero_grad()  # 清空之前的梯度

        # 计算模型输出
        results = model(image_tensor, verbose=False)
        detections = results[0].boxes.data

        # 找到具有最大置信度的目标
        max_det = max(detections, key=lambda x: x[4])
        max_conf = max_det[4].item()  # 置信度转换为 float 类型

        # 计算损失并执行反向传播
        loss = -max_conf  # 使用负的最大置信度作为损失函数
        loss.backward()  # 执行反向传播

        # 获取图片的梯度
        grad = image_tensor.grad.cpu().numpy()
        perturbation += EPSILON * grad[0].transpose(1, 2, 0)  # 转回为原图形状
        perturbation = np.clip(perturbation, -20, 20)

    # 保存攻击后的图片
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "attacked_0001_pgd.jpg"), image)

    # 计算L2距离
    l2_dist = compute_l2_distance(orig_image, image)
    print(f"L2 distance: {l2_dist:.2f}")

if __name__ == "__main__":
    latency_attack_pgd()
