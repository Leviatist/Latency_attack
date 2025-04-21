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