from boxes import get_boxes_info
from config import IMAGE_PATH

boxes = get_boxes_info(IMAGE_PATH)
for i,box in enumerate(boxes):
    print(i,box)