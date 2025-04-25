import torch
import cv2
import numpy as np
from ultralytics import YOLO
from config import MODELV8N_PATH, IMAGE_PATH, OUTPUT_PATH
from getBoxes import get_boxes_info  # 导入你提供的get_boxes_info方法

# 加载YOLO模型
model = YOLO(MODELV8N_PATH)

# 读取图像
image = cv2.imread(IMAGE_PATH)  # 修改为你的图像路径

# 调整图像大小，使其适应YOLO的输入要求 (640x640)
image_resized = cv2.resize(image, (640, 640))  # 调整图像大小
image_tensor = torch.tensor(image_resized).float().permute(2, 0, 1) / 255.0  # 归一化，CHW格式
image_tensor.requires_grad = True  # 允许计算梯度

# PGD攻击参数
EPSILON = 0.03  # 每步扰动的最大值
ATTACK_ITER = 40  # 攻击的迭代次数
CONF_THRESHOLD = 0.8  # 置信度阈值

# 获取预测框信息
boxes_info = get_boxes_info(IMAGE_PATH)

# 目标框的类别和置信度最大化
def pgd_attack(image_tensor, model, boxes_info, epsilon=EPSILON, attack_iter=ATTACK_ITER):
    """
    使用PGD攻击最大化每个框的置信度
    """
    # 获取目标框的类别和置信度
    target_class = boxes_info[0]["class_id"]  # 这里只取第一个框的类别作为目标，你可以扩展为多个框
    target_conf = boxes_info[0]["conf"]
    
    # 将模型输入到图像
    for _ in range(attack_iter):
        loss=0
        
        # 进行前向传播，获得预测结果
        results = model(image_tensor.unsqueeze(0),conf=0.0001)[0]
        print("image_tensor.grad_fun:",image_tensor.grad_fn)

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = results.names[cls]
            conf = torch.tensor(conf, dtype=torch.float32)
            print("conf:",conf.grad_fn)

            loss -=  -torch.log(conf)

        # 反向传播计算梯度
        image_tensor.grad = None  # 清除以前的梯度
        loss.backward()

        # 使用PGD更新图像
        with torch.no_grad():
            # 给图像添加扰动
            image_tensor += epsilon * image_tensor.grad.sign()
            
            # 限制扰动的范围，使图像保持在合法的像素范围内
            image_tensor = torch.clamp(image_tensor, 0, 1)

    return image_tensor

# 进行PGD攻击
attacked_image_tensor = pgd_attack(image_tensor, model, boxes_info)

# 将攻击后的图像转换为标准格式并保存
attacked_image = attacked_image_tensor.cpu().numpy() * 255  # 转回HWC格式
attacked_image = attacked_image.astype(np.uint8)

cv2.imwrite(OUTPUT_PATH, attacked_image)  # 保存攻击后的图像
