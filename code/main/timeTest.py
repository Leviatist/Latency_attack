import time
import cv2
from lib.detect import detect_image
from lib.config import DUMMYIMG_PATH, ORIGINIMG_PATH, ATKEDIMG_PATH

def test_detection_time(image_path, tag, rounds=10):
    image = cv2.imread(image_path)
    total_time = 0.0
    for _ in range(rounds):
        start = time.time()
        _ = detect_image(image)
        end = time.time()
        total_time += (end - start)
    avg_time_ms = (total_time / rounds) * 1000
    print(f"{tag} Detection Time ({rounds} rounds avg): {avg_time_ms:.2f} ms")

if __name__ == "__main__":
    rounds = 100  # 轮次，可以自由改
    print("检测耗时对比：")
    test_detection_time(DUMMYIMG_PATH, "预热图", rounds)
    test_detection_time(ATKEDIMG_PATH, "攻击图", rounds)
    test_detection_time(ORIGINIMG_PATH, "原始图", rounds)