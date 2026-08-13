import os
import cv2
import numpy as np
from PIL import Image

orig_path = r"C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\image.png"
edit_path = r"C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\image_edited_test.jpg"

for label, path in [("Original", orig_path), ("Edited", edit_path)]:
    if not os.path.exists(path):
        print(f"{label} missing")
        continue
    
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Bimodal overlay test (High contrast step transitions)
    # Binary contrast ratio (count of pixels near 0 and near 255 in 32x32 local windows)
    h, w = gray.shape[:2]
    max_bimodal_score = 0
    
    for y in range(0, h - 32, 16):
        for x in range(0, w - 32, 16):
            patch = gray[y:y+32, x:x+32]
            p_min, p_max = np.min(patch), np.max(patch)
            if (p_max - p_min) > 200:  # Absolute black to white contrast
                # Count pixels near 0 and near 255
                black_cnt = np.sum(patch < 40)
                white_cnt = np.sum(patch > 215)
                if black_cnt > 15 and white_cnt > 15:
                    bimodal_score = black_cnt + white_cnt
                    if bimodal_score > max_bimodal_score:
                        max_bimodal_score = bimodal_score
                        
    print(f"[{label}] Max Bimodal Text Overlay Score: {max_bimodal_score}")
