import torch
from torchvision import transforms
from PIL import Image
from config import MODELV8N_PATH, IMAGE_PATH
from ultralytics import YOLO

# 假设你已经有一个训练好的模型
model = YOLO(MODELV8N_PATH)  # 载入你的模型

# 设置为评估模式
model.eval()

# 加载图片并预处理
image = Image.open(IMAGE_PATH).convert('RGB')

# 定义图像预处理的转换操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正则化
])

input_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度

# 将输入张量移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
model = model.to(device)

# 允许计算梯度
input_tensor.requires_grad_()

# 前向传播：获取模型的输出
output = model(input_tensor)

# 打印模型的原始输出
print("未经后处理的模型输出:", output)
print(output[0].boxes.conf.grad_fn)