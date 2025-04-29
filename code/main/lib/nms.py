import torch
import time
from torchvision.ops import nms

# 假设 preds 是 [8400, 84] 的 tensor
def run_nms_on_preds(preds, iou_threshold=0.5):
    # 取出位置参数
    xywh = preds[:, :4]  # (8400, 4)

    # 把 (x, y, w, h) 转成 (x1, y1, x2, y2)
    xy = xywh[:, :2]
    wh = xywh[:, 2:]
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)  # (x1, y1, x2, y2)

    # 取出最大类别置信度作为分数
    class_confs = preds[:, 4:]  # (8400, 80)
    scores, labels = torch.max(class_confs, dim=1)  # scores: (8400,)

    # 开始计时
    start_time = time.perf_counter()

    # 执行 NMS
    kept_indices = nms(boxes, scores, iou_threshold)

    # 结束计时
    end_time = time.perf_counter()

    print(f"NMS耗时: {end_time - start_time:.6f} 秒")
    print(f"保留了 {kept_indices.numel()} 个框")
    
    return kept_indices