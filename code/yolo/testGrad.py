import torch
from config import IMAGE_PATH,CSV_PATH
from grad import get_raw_output,predsToCsv

# 拿到模型原始输出
preds, im_tensor = get_raw_output(IMAGE_PATH)

predsToCsv(preds,CSV_PATH)

'''
# 检查 preds 和 im_tensor 是否有梯度
print("preds requires_grad:", preds.requires_grad)
print("im_tensor requires_grad:", im_tensor.requires_grad)

# 查看 preds 的形状
print("preds shape:", preds.shape)

# 随便看一个预测结果
sample_pred = preds[0, 0]  # 第一个anchor的预测
print("Sample pred:", sample_pred)
print("x, y, w, h, objectness, classes:", sample_pred[:4], sample_pred[4], sample_pred[5:])

# 检查是否能反向传播
loss = preds.sum()  # 随便造一个假的loss
loss.backward()
print("im_tensor grad exists:", im_tensor.grad is not None)
'''