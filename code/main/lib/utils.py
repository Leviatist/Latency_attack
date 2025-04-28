import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils

def save_image_from_tensor(tensor, file_path, use_pil=True):
    """
    将 PyTorch 张量保存为图像文件。
    
    参数：
    - tensor: 要保存的 PyTorch 张量 (shape: [C, H, W] 或 [B, C, H, W])。
    - file_path: 保存图像的文件路径（例如 'image.png'）。
    - use_pil: 是否使用 PIL 库保存图像，默认为 True。如果为 False，则使用 torchvision.utils.save_image。
    """
    # 确保图像在 0-1 范围内
    tensor = torch.clamp(tensor, 0, 1)
    
    if use_pil:
        # 转换为 PIL 图像
        to_pil = transforms.ToPILImage()
        if tensor.dim() == 4:  # 如果是批次图像 (B, C, H, W)，需要去除批次维度
            tensor = tensor.squeeze(0)
        pil_img = to_pil(tensor)
        pil_img.save(file_path)
    else:
        # 使用 torchvision 保存图像
        vutils.save_image(tensor, file_path)

    print(f"图像已保存: {file_path}")

