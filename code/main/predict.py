import cv2 
from ultralytics import YOLO
from lib.config import MODELV8N_PATH, ORIGINIMG_PATH, ATKEDIMG_PATH, PREDICTEDIMG_PATH

model = YOLO(MODELV8N_PATH)
image1 = cv2.imread(ORIGINIMG_PATH)
image2 = cv2.imread(ATKEDIMG_PATH)

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
result_img1, _, labels1 = predict_and_detect(model, image1, conf=0.3)
result_img2, _, labels2 = predict_and_detect(model, image2, conf=0.3)
# 输出检测到的类型标签
<<<<<<< HEAD
print("Detected Labels:", labels)
# 显示带标注的图片
cv2.imshow("Image", result_img)
'''

cv2.waitKey(0)
=======
print("ORIGIN Detected Labels:", labels1)
print("ATKED Detected Labels:", labels2)

cv2.imwrite(PREDICTEDIMG_PATH, result_img2)

'''
# 显示带标注的图片
cv2.imshow("Image", result_img)
>>>>>>> ea5e834b017452ffe144040ef517c6754713d3ca
'''