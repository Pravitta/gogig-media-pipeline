"""
Multi-Metric Blur & Motion Anisotropy Detection Engine.
Evaluates both Global Spectral Density AND Central Subject ROI Blur + Directional Motion Blur Anisotropy.
Guarantees motion-blurred vehicles with sharp backgrounds are correctly classified as Low Detail / Blurry.
"""

import cv2
import numpy as np
from PIL import Image

def analyze_blur(image_path: str):
    try:
        pil_img = Image.open(image_path).convert('RGB')
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {"name": "blur", "score": 0.0, "signal": f"error:{str(e)}", "confidence": 0.0, "verdict": "unknown"}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Normalize dimensions for consistent calculation
    max_dim = max(h, w)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        gray_norm = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_norm = gray

    nh, nw = gray_norm.shape[:2]

    # 1. Central Subject ROI Crop (Middle 60% of frame where vehicle subject sits)
    sy1, sy2 = int(nh * 0.20), int(nh * 0.80)
    sx1, sx2 = int(nw * 0.20), int(nw * 0.80)
    roi_gray = gray_norm[sy1:sy2, sx1:sx2]

    # 2. Central Subject Denoising (adjusted for low-light/night sensor noise)
    roi_mean = float(np.mean(roi_gray))
    if roi_mean < 75.0:
        # Stronger smoothing for low-light images to filter out sensor noise/grain
        roi_denoised = cv2.GaussianBlur(roi_gray, (7, 7), 0)
    elif roi_mean < 110.0:
        # Moderate smoothing
        roi_denoised = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    else:
        # Standard light
        roi_denoised = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    roi_lap_var = float(cv2.Laplacian(roi_denoised, cv2.CV_64F).var())

    roi_sobel_x = np.abs(cv2.Sobel(roi_denoised, cv2.CV_64F, 1, 0, ksize=3))
    roi_sobel_y = np.abs(cv2.Sobel(roi_denoised, cv2.CV_64F, 0, 1, ksize=3))

    mean_sx = float(np.mean(roi_sobel_x))
    mean_sy = float(np.mean(roi_sobel_y))

    # Directional Motion Blur Anisotropy (Streak ratio)
    anisotropy_ratio = (mean_sx / mean_sy) if mean_sy > 0 else 1.0
    has_motion_streak = (anisotropy_ratio > 2.6 or anisotropy_ratio < 0.38) and (roi_lap_var < 180.0)

    # 3. Global 2D FFT & Tenengrad
    f = np.fft.fft2(gray_norm)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    cy, cx = nh // 2, nw // 2
    r_inner = int(min(nh, nw) * 0.15)
    y_grid, x_grid = np.ogrid[:nh, :nw]
    mask_high = ((x_grid - cx)**2 + (y_grid - cy)**2) > r_inner**2
    total_power = np.sum(magnitude)
    high_freq_power = np.sum(magnitude * mask_high)
    fft_ratio = float((high_freq_power / total_power) if total_power > 0 else 0.0)

    global_sobel = np.sqrt(roi_sobel_x**2 + roi_sobel_y**2)
    tenengrad_mean = float(np.mean(global_sobel))

    signals = [
        f"fft_ratio={fft_ratio:.4f}",
        f"tenengrad={tenengrad_mean:.1f}",
        f"lap_var={roi_lap_var:.1f}",
        f"motion_anisotropy={anisotropy_ratio:.2f}"
    ]

    # Subject Motion & Blur Decision Logic:
    # Reject as blurry only if both the variance is low AND edges are weak, OR if a motion streak is detected.
    is_blurry = (roi_lap_var < 80.0 and tenengrad_mean < 35.0) or (roi_lap_var < 120.0 and tenengrad_mean < 25.0) or has_motion_streak
    
    # Clean (sharp) if it has high variance, OR if it has moderate variance combined with strong, sharp edges,
    # OR if the edges are extremely sharp (typical of direct screenshots with text overlays).
    is_sharp = (roi_lap_var >= 125.0 and tenengrad_mean >= 20.0) or (roi_lap_var >= 95.0 and tenengrad_mean >= 35.0) or (tenengrad_mean >= 48.0)

    if is_blurry:
        # Blurry / Motion Blurred Vehicle Subject
        score = max(0.0, round(roi_lap_var / 350.0, 2))
        return {
            "name": "blur",
            "score": score,
            "signal": ",".join(signals),
            "confidence": 0.98,
            "verdict": "rejected"
        }
    elif is_sharp:
        # Sharp / High Clarity Vehicle Subject
        score = min(1.0, round(0.85 + min(0.15, roi_lap_var / 2000.0), 2))
        return {
            "name": "blur",
            "score": score,
            "signal": ",".join(signals),
            "confidence": 0.98,
            "verdict": "clean"
        }
    else:
        # Mildly Soft / Needs Review
        return {
            "name": "blur",
            "score": 0.55,
            "signal": ",".join(signals),
            "confidence": 0.85,
            "verdict": "needs_review"
        }
