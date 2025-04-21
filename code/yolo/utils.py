import numpy as np
import cv2

def split_image(image, grid_size):
    h, w, _ = image.shape
    grids = []
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            x2, y2 = min(x+grid_size, w), min(y+grid_size, h)
            cell = image[y:y2, x:x2].copy()
            grids.append((x, y, x2, y2, cell))
    return grids

def compute_l2_distance(img1, img2):
    diff = (img1.astype(np.float32) - img2.astype(np.float32)).flatten()
    return np.linalg.norm(diff)
