# Latency attack
## 目前暂定的思路和难点
暂定的思路：
+ 我先对我的图片进行正向传播，获得Boxes位置信息和对不同类别的概率。
+ 然后我将Boxes信息加某个类别作为标签，计算loss，通过PGD攻击不断提高某个类别的置信度，直到该置信度达到阈值
## 项目背景简介
我有一个项目，要实现latency_attack
项目文件夹内有文件夹code,data,model,docs
 code内有两个文件夹
+ env内是cleanup.sh，setup.sh，是环境建立相关的代码
+ yolo内有若干个文件
    - attack.py 实现pgd攻击
    - config.py 项目的各种常量信息
    - detect.py 对图片进行目标检测的函数
    - getinfo.py 获得环境信息
    - predict.py 对图片目标进行预测
    - time_test.py 检测延时攻击的效果
    - utils.py一些工具函数
    - getBoxes.py 获取图像预测框信息，置信度信息

data/img目录下有测试用的图片
model 里面是用来放yolo11.pt模型的
## 详细代码
### attack.py
```python
import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from config import MODEL_PATH,IMAGE_PATH, OUTPUT_PATH, CONF_THRESHOLD, GRID_SIZE, LAMBDA, ATTACK_ITER, EPSILON
```
### config.py
```python
import os

abs_path = os.path.abspath(__file__)
pro_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))

MODEL_PATH = os.path.join(pro_path, 'model/yolo11x.pt')
IMAGE_PATH = os.path.join(pro_path, 'data/Img/0001.jpg')
OUTPUT_PATH = os.path.join(pro_path, 'output/')

CONF_THRESHOLD = 0.8
GRID_SIZE = 30
LAMBDA = 0.01
ATTACK_ITER = 1000
EPSILON = 3
```
### detect.py
```python
from ultralytics import YOLO

from config import MODEL_PATH

model = YOLO(MODEL_PATH)

def detect_image(image):
    results = model(image, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2, conf = box.xyxy[0].tolist() + [box.conf[0].item()]
            detections.append([x1, y1, x2, y2, conf])
    return detections
```

### predict.py
```python
import cv2 
from ultralytics import YOLO
from config import MODEL_PATH, IMAGE_PATH, OUTPUT_PATH

model = YOLO(MODEL_PATH)
image_0 = cv2.imread(IMAGE_PATH)
image_1 = cv2.imread(OUTPUT_PATH+ "attacked_0001_pgd.jpg")

def predict(chosen_model, img, classes=[], conf=0.3):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    detected_labels = []  # 用于保存检测到的标签

    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            detected_labels.append(label)  # 将检测到的标签添加到列表中

            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, label,
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)

    return img, results, detected_labels  # 返回检测到的标签列表

# 读取图片
result_img, _, labels = predict_and_detect(model, image_1, conf=0.3)

# 显示带标注的图片
cv2.imshow("Image", result_img)
cv2.imwrite("Project/Active/Latency_attack/data/Result", result_img)

# 输出检测到的类型标签
print("Detected Labels:", labels)

cv2.waitKey(0)
```
### time_test.py
```python
import time
import cv2
from detect import detect_image
from config import IMAGE_PATH, OUTPUT_PATH

def test_detection_time(image_path, tag):
    image = cv2.imread(image_path)
    start = time.time()
    _ = detect_image(image)
    end = time.time()
    print(f"{tag} Detection Time: {(end-start)*1000:.2f} ms")

if __name__ == "__main__":
    print("检测耗时对比：")
    test_detection_time(OUTPUT_PATH + "attacked_0001_pgd.jpg", "攻击图")
    test_detection_time(IMAGE_PATH, "原图")
```
### utils.py
```python
import numpy as np
import cv2

def compute_l2_distance(img1, img2):
    diff = (img1.astype(np.float32) - img2.astype(np.float32)).flatten()
    return np.linalg.norm(diff)
```
### getBoxes.py
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
> getBoxes使用方法
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
一步步实现我的思路，首先我需要我给出一个Loss，获取图像对它的梯度