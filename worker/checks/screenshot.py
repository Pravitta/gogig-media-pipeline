"""
Screenshot & Screen Subpixel Moiré Classifier.
Distinguishes actual digital screenshots and re-photographed screens from natural 
camera photography (including night street scenes with lights and signs).
"""

import cv2
import numpy as np
from PIL import Image

def analyze_screenshot(image_path: str):
    try:
        score = 0.0
        signals = []
        
        # 1. EXIF Metadata Software Check
        img = Image.open(image_path)
        exif = img.getexif()
        has_exif = bool(exif)
        
        if not has_exif:
            score += 0.05
            signals.append("missing_exif")
        else:
            software = str(exif.get(305, "")).lower()
            if any(x in software for x in ["screenshot", "screen", "capture", "snipping"]):
                score += 0.70
                signals.append("exif_software_screenshot")
        
        # 2. Aspect Ratio Match
        width, height = img.size
        if min(width, height) > 0:
            ratio = max(width, height) / min(width, height)
            # Modern smartphone displays (19.5:9, 20:9, 21:9) range from 2.15 to 2.35
            common_screen_ratios = [1.777, 1.6, 2.166, 2.222, 1.333, 2.23]
            is_screen_ratio = any(abs(ratio - r) < 0.04 for r in common_screen_ratios)
            if is_screen_ratio:
                score += 0.35
                signals.append(f"screen_aspect_ratio_{ratio:.2f}")

        # 3. Screen Grid Moiré Verification (Orthogonal Subpixel Lattice Test)
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            gray_norm = cv2.resize(gray, (256, 256))
            f = np.fft.fft2(gray_norm)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            cy, cx = 128, 128
            r = 20
            y, x = np.ogrid[:256, :256]
            mask = ((x - cx)**2 + (y - cy)**2) > r**2
            
            high_freq = magnitude * mask
            max_peak = np.max(high_freq)
            mean_val = np.mean(high_freq)
            
            pmr = max_peak / mean_val if mean_val > 0 else 0.0
            
            # Check for orthogonal grid symmetry (true screen subpixel grid has 4 symmetric frequency peaks)
            # Find coordinates of top 10 peaks
            flat_indices = np.argsort(high_freq.ravel())[-10:]
            peaks_y, peaks_x = np.unravel_index(flat_indices, (256, 256))
            
            # Count peaks aligned on horizontal/vertical frequency axes
            axis_peaks = 0
            for py, px in zip(peaks_y, peaks_x):
                if abs(py - cy) < 4 or abs(px - cx) < 4:
                    axis_peaks += 1

            # Check if image has a solid studio/cutout background (white or black canvas)
            border_top = gray[:15, :]
            border_bottom = gray[-15:, :]
            border_left = gray[:, :15]
            border_right = gray[:, -15:]
            border_pixels = np.concatenate([border_top.ravel(), border_bottom.ravel(), border_left.ravel(), border_right.ravel()])
            is_solid_canvas = (np.mean(border_pixels > 245) > 0.50) or (np.mean(border_pixels < 10) > 0.50)

            # Only flag Moiré if PMR is extremely high (> 45.0), orthogonal axis peaks >= 4, AND NOT a solid canvas
            if pmr > 45.0 and axis_peaks >= 4 and not is_solid_canvas:
                score += 0.70
                signals.append(f"fft_screen_grid_pmr_{pmr:.1f}")

        # 4. Direct Digital UI Text Detection (Home/Lock screen indicators)
        try:
            import pytesseract
            ocr_text = pytesseract.image_to_string(img).strip().lower()
            ui_keywords = ["whatsapp", "play store", "files", "chrome", "cleaner", "calendar", "tools", "google", "vo", "wifi", "battery"]
            detected_ui = [kw for kw in ui_keywords if kw in ocr_text]
            if len(detected_ui) >= 2:
                score += 0.80
                signals.append(f"ui_lockscreen_detected({','.join(detected_ui[:3])})")
        except Exception:
            pass

        final_score = min(1.0, max(0.0, score))
        verdict = "needs_review" if final_score >= 0.60 else "clean"
        
        return {
            "name": "screenshot",
            "score": round(final_score, 2),
            "signal": ",".join(signals) if signals else "natural_camera",
            "confidence": 0.90,
            "verdict": verdict
        }
        
    except Exception as e:
        return {
            "name": "screenshot", 
            "score": 0.0, 
            "signal": f"error:{str(e)}", 
            "confidence": 0.0, 
            "verdict": "unknown"
        }
