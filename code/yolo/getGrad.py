import torch
from ultralytics import YOLO
from config import MODEL_PATH, IMAGE_PATH

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(MODEL_PATH).to(device)

# 读取并预处理图片
def preprocess_image(image_path, img_size=640):
    import cv2
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    resized_image = cv2.resize(image, (new_w, new_h))

    image_tensor = torch.from_numpy(resized_image).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return image_tensor / 255.0

def get_gradient(image_path):
    # 获取图片并准备输入
    image_tensor = preprocess_image(image_path)
    
    # 确保计算梯度
    image_tensor.requires_grad_(True)
    
    # 前向传播
    results = model(image_tensor)
    
    # 获取第一个结果的置信度 (confidence) 并计算损失
    # 注意：results[0].boxes 是一个包含所有检测框的对象
    boxes = results[0].boxes
    if len(boxes) > 0:
        # 假设我们使用第一个检测框的最大置信度
        confidence = boxes.conf.max()  # 最大置信度
    else:
        # 如果没有检测到物体，可以设置一个默认值
        confidence = torch.tensor(0.0, device=device)

    # 因为 confidence 本身不要求梯度，我们需要保证其依赖的张量要求梯度
    # 在这里，`image_tensor` 已经要求了梯度
    loss = confidence  # 直接将 confidence 作为损失函数
    
    # 清空之前的梯度
    model.zero_grad()  
    
    # 计算损失的梯度
    loss.backward()
    
    # 获取梯度
    gradients = image_tensor.grad
    
    return gradients

if __name__ == "__main__":
    gradients = get_gradient(IMAGE_PATH)
    print(f"Calculated gradients: {gradients.shape}")
