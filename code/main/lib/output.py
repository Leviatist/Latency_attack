import torch
import cv2
import pandas as pd
from ultralytics import YOLO
from lib.config import MODELV8N_PATH

# 加载模型
model = YOLO(MODELV8N_PATH)

def simple_preprocess(image):
    # resize到640×640，并归一化
    image = cv2.resize(image, (640, 640))  # Resize to model's expected input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H,W,C) -> (C,H,W)
    image = image / 255.0  # normalize to 0-1
    return image

def get_raw_output_img(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    im_tensor = simple_preprocess(image)  # 手动处理成tensor

    im_tensor = im_tensor.unsqueeze(0)  # 加上batch维度 [1, 3, 640, 640]
    im_tensor.requires_grad_(True)  # 开启梯度

    with torch.set_grad_enabled(True):
        preds = model.model(im_tensor)[0]  # 手动forward
    
    # 先 squeeze 去掉 batch 维度
    preds = preds.squeeze(0)  # 现在 shape 是 [84, 8400]

    # 转置一下，让每一行是一个预测点
    preds = preds.permute(1, 0)  # 现在 shape 是 [8400, 84]

    return preds, im_tensor

def get_raw_output_tensor(im_tensor):
    im_tensor.requires_grad_(True)  # 开启梯度

    with torch.set_grad_enabled(True):
        preds = model.model(im_tensor)[0]  # 手动forward
    
    # 先 squeeze 去掉 batch 维度
    preds = preds.squeeze(0)  # 现在 shape 是 [84, 8400]

    # 转置一下，让每一行是一个预测点
    preds = preds.permute(1, 0)  # 现在 shape 是 [8400, 84]

    return preds, im_tensor

def predsToCsv(preds,csvPath):
    # 生成列名
    columns = ['x', 'y', 'w', 'h'] + [f'conf{i}' for i in range(1, 81)]

    # 转成 pandas dataframe
    df = pd.DataFrame(preds.detach().cpu().numpy(), columns=columns)

    # 保存到本地，或者直接使用
    df.to_csv(csvPath, index=False)

    print(df.head())