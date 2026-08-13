import os
import cv2
import numpy as np
from PIL import Image

original_path = r"C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\image.png"
edited_path = r"C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\image_edited_test.jpg"

for name, path in [("Original", original_path), ("Edited", edited_path)]:
    if not os.path.exists(path):
        print(f"{name} file not found!")
        continue
        
    pil_img = Image.open(path).convert('RGB')
    h, w = pil_img.size
    max_dim = max(h, w)
    if max_dim > 1000:
        scale = 1000.0 / max_dim
        resized_img = pil_img.resize((int(h * scale), int(w * scale)), Image.Resampling.LANCZOS)
    else:
        resized_img = pil_img
        
    gray = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2GRAY)
    
    # Let's test with a 3x3 median filter which preserves edges but removes point noise
    smoothed = cv2.medianBlur(gray, 3)
    
    sx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx**2 + sy**2)
    
    # Try different thresholds (e.g. 180, 200, 220)
    for thr in [180, 200, 220]:
        sharp_edges = grad_mag > thr
        sharp_pixel_count = np.sum(sharp_edges)
        
        # Grid block ratio
        grid_h, grid_w = smoothed.shape[:2]
        block_h, block_w = grid_h // 5, grid_w // 5
        y_indices, x_indices = np.where(sharp_edges)
        
        block_counts = []
        for row in range(5):
            for col in range(5):
                y1, y2 = row * block_h, (row + 1) * block_h
                x1, x2 = col * block_w, (col + 1) * block_w
                count = np.sum((y_indices >= y1) & (y_indices < y2) & (x_indices >= x1) & (x_indices < x2))
                block_counts.append(count)
        
        max_block_count = max(block_counts) if block_counts else 0
        max_block_ratio = max_block_count / sharp_pixel_count if sharp_pixel_count > 0 else 0.0
        
        print(f"[{name}] Threshold {thr}: Count = {sharp_pixel_count}, Max Block Ratio = {max_block_ratio:.2f}")

    # Run ELA diagnostic
    import io
    from PIL import ImageChops, ImageEnhance
    try:
        pil_img = Image.open(path).convert('RGB')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        ela_img = ImageChops.difference(pil_img, resaved)
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        enhancer = ImageEnhance.Brightness(ela_img)
        ela_enhanced = enhancer.enhance(10.0)
        arr = np.array(ela_enhanced)
        std_dev = np.std(arr)
        print(f"[{name}] ELA Max Diff: {max_diff}, ELA Std Dev: {std_dev:.2f}")
    except Exception as e:
        print(f"[{name}] ELA error: {e}")

