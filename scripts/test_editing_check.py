"""
Test Script: test_editing_check.py
Demonstrates how the editing detection engine detects image manipulation.
1. Takes the user's original image
2. Splices a simulated overlay onto it and saves as an edited version
3. Runs ELA (Error Level Analysis) and prints results side-by-side
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from worker.checks.editing import analyze_editing

def run_test():
    original_path = r"C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\image.png"
    edited_path = r"C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\image_edited_test.jpg"

    if not os.path.exists(original_path):
        print(f"Original image not found at {original_path}")
        return

    # 1. Establish initial compression history by saving background as 65% JPEG first
    img = cv2.imread(original_path)
    temp_path = r"C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\temp_65.jpg"
    cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 65])
    
    # Load the compressed background, and splice the mock plate
    spliced_img = cv2.imread(temp_path)
    h, w = spliced_img.shape[:2]
    
    # Draw a high-contrast white rectangle (spliced edit) in the center
    cv2.rectangle(spliced_img, (w//4, h//2), (3*w//4, h//2 + h//8), (255, 255, 255), -1)
    cv2.putText(spliced_img, "EDITED_PLATE", (w//4 + 20, h//2 + h//16), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    # Save the final image at 90% JPEG quality (creating a double-compression layer mismatch)
    cv2.imwrite(edited_path, spliced_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
            
    print(f"Generated double-compressed edited image for test at: {edited_path}\n")

    # 2. Analyze both images using ELA and Sharp Edges
    print("--- Running Editing Detection Check ---")
    
    print("\n[Original Image]")
    orig_res = analyze_editing(original_path)
    print(f"Verdict: {orig_res['verdict']}")
    print(f"Score: {orig_res['score']}")
    print(f"Signals: {orig_res['signal']}")

    print("\n[Edited Image (Spliced Plate)]")
    edit_res = analyze_editing(edited_path)
    print(f"Verdict: {edit_res['verdict']}")
    print(f"Score: {edit_res['score']}")
    print(f"Signals: {edit_res['signal']}")

    print("\nFile kept at path for manual dashboard testing!")

if __name__ == "__main__":
    run_test()
