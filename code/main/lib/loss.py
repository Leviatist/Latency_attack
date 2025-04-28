import torch

def confidence_increase_loss(preds):
    """
    preds: tensor of shape [8400, 84]
    """
    class_preds = preds[:, 4:]  # 取80维类别预测
    class_probs = torch.sigmoid(class_preds)  # 通常需要过sigmoid，保证在0~1之间
    max_probs_per_box, _ = class_probs.max(dim=1)  # 每个框最大类别置信度
    loss = max_probs_per_box.mean()  # 取平均后加负号
    return loss
