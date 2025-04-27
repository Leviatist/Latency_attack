# Latency attack
## 目前暂定的思路和难点
暂定的思路：
+ 我先对我的图片进行正向传播，获得Boxes位置信息和对不同类别的概率。
+ 然后我将Boxes信息加某个类别作为标签，计算loss，通过PGD攻击不断提高每个box内某个类别的置信度，直到该置信度达到阈值
## 项目背景简介
我有一个项目，要实现latency_attack
项目文件夹内有文件夹code,data,model,docs
 code内有两个文件夹
+ env内是cleanup.sh，setup.sh，是环境建立相关的代码
+ yolo内有若干个文件
    - config.py 项目的各种常量信息
    - boxes.py 获取图像预测框信息，置信度信息
    - testBoxes.py boxes的使用案例

data/img目录下有测试用的图片
model 里面是用来放模型的
## 详细代码
### boxes.py
```python
from ultralytics import YOLO
import cv2
from config import MODELV8N_PATH

# 加载模型
model = YOLO(MODELV8N_PATH)

def get_boxes_info(image_path, conf_threshold=0):
    image = cv2.imread(image_path)
    results = model(image, verbose=False, conf=0.0001)[0]

    boxes_info = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = results.names[cls]

        if conf >= 0:
            boxes_info.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "class_id": cls,
                "label": label
            })

    return boxes_info
```
> testBoxes.py
```python
from ultralytics import YOLO
import cv2
from config import MODELV8N_PATH

# 加载模型
model = YOLO(MODELV8N_PATH)

def get_boxes_info(image_path, conf_threshold=0):
    image = cv2.imread(image_path)
    results = model(image, verbose=False, conf=0.0001)[0]

    boxes_info = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = results.names[cls]

        if conf >= 0:
            boxes_info.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "class_id": cls,
                "label": label
            })

    return boxes_info
```
### 诉求
一步步实现我的思路，首先我需要获得YOLO的原始输出，保留梯度信息的输出