import cv2
import numpy as np
from PIL import Image

def analyze_brightness(image_path: str):
    """
    Analyzes the brightness of an image by computing the mean pixel intensity 
    after converting it to grayscale.
    """
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return {
            "name": "brightness", 
            "score": 0.0, 
            "signal": "error_loading_image", 
            "confidence": 0.0, 
            "verdict": "unknown"
        }
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()
    
    # Threshold Explanation: 
    # An 8-bit grayscale image has pixel values ranging from 0 (pure black) 
    # to 255 (pure white). A mean intensity below 50 means the image is 
    # heavily skewed towards black and is generally considered very dark/underexposed.
    threshold = 50.0 
    is_low_light = bool(mean_intensity < threshold)
    
    if is_low_light:
        # Scale score from 0 to 1 based on how close it is to the threshold
        score = max(0.0, mean_intensity / threshold)
        verdict = "needs_review"
    else:
        score = 1.0
        verdict = "clean"
        
    return {
        "name": "brightness",
        "score": round(score, 2),
        "signal": f"mean_intensity={mean_intensity:.1f}",
        "low_light": is_low_light,
        "threshold_used": threshold,
        "threshold_explanation": "Mean pixel intensity < 50 on a 0-255 scale indicates significant underexposure.",
        "confidence": 0.9,
        "verdict": verdict
    }
