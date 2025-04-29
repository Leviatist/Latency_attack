import torch
from lib.config import IMAGE_PATH, OUTPUT_PATH
from lib.output import get_raw_output_img, get_raw_output_tensor
from lib.loss import confidence_increase_loss
from lib.utils import save_image_from_tensor

# 配置PGD攻击参数
epsilon = 8 / 255  # 最大扰动
alpha = 2 / 255    # 步长
num_steps = 200   # 总步数

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 读取原始图像和预测
preds, im_tensor = get_raw_output_img(IMAGE_PATH)

# 保存原始图像备份
original_im_tensor = im_tensor.clone().detach()

# 确保图像可以求梯度
im_tensor.requires_grad = True
im_tensor.retain_grad()  # 保证在反向传播时保留梯度

# PGD攻击主循环
for step in range(num_steps):
    # 1. 预测
    preds, im_tensor = get_raw_output_tensor(im_tensor) 

    # 确保图像可以求梯度
    im_tensor.retain_grad()  # 保证在反向传播时保留梯度

    # 2. 计算Loss
    loss = confidence_increase_loss(preds)

    # 3. 反向传播
    if im_tensor.grad is not None:
        im_tensor.grad.zero_()
        
    loss.backward(retain_graph=True)

    # 4. 取梯度
    grad = im_tensor.grad

    print("im_tensor grad exists:", im_tensor.grad is not None)

    # 5. 更新图像（梯度上升，max置信度）
    im_tensor = im_tensor + alpha * grad.sign()

    # 6. 投影到合法范围 [original - epsilon, original + epsilon]
    im_tensor = torch.max(torch.min(im_tensor, original_im_tensor + epsilon), original_im_tensor - epsilon)
    im_tensor = torch.clamp(im_tensor, 0, 1)  # 同时也保证在0-1区间

    # 7. 准备下一步
    im_tensor.requires_grad_(True)

    print(f"Step {step+1}/{num_steps} done, loss: {loss.item():.4f}")

# 攻击完成，im_tensor 就是最终的攻击图了
final_attack_img = im_tensor.detach()

# 你可以保存 final_attack_img，或者直接用它去做后续检测
save_image_from_tensor(final_attack_img, OUTPUT_PATH)