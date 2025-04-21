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
