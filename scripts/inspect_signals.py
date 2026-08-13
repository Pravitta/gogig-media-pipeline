"""
Instrumentation Script: inspect_signals.py
Extracts raw numerical signals and OCR pipeline traces across sample images
without applying verdict/threshold logic.
"""

import os
import sys
import glob
import cv2
import re
import shutil
import pytesseract
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

# Ensure Tesseract executable path on Windows host
if not shutil.which("tesseract"):
    for p in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break


def inspect_image_signals(image_path: str):
    results = {"filename": os.path.basename(image_path)}

    try:
        pil_img = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
    except Exception as e:
        results["error"] = str(e)
        return results

    # 1. Dimensions & Aspect Ratio
    megapixels = (w * h) / 1000000.0
    aspect_ratio = max(w, h) / float(min(w, h)) if min(w, h) > 0 else 1.0
    results["dimensions"] = {
        "width": w,
        "height": h,
        "megapixels": round(megapixels, 2),
        "aspect_ratio": round(aspect_ratio, 2),
    }

    # 2. Blur: Raw Laplacian Variance (Unscaled & Normalized to 1000px)
    raw_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    max_dim = max(h, w)
    if max_dim > 1000:
        scale = 1000.0 / max_dim
        norm_gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        norm_gray = gray
    norm_variance = cv2.Laplacian(norm_gray, cv2.CV_64F).var()

    results["blur"] = {
        "raw_laplacian_variance": round(raw_variance, 2),
        "normalized_1000px_variance": round(norm_variance, 2),
    }

    # 3. Brightness & Contrast Intensity
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    dark_ratio = np.sum(gray < 40) / float(gray.size)
    overexposed_ratio = np.sum(gray > 220) / float(gray.size)

    results["brightness"] = {
        "mean_intensity": round(mean_intensity, 2),
        "std_intensity": round(std_intensity, 2),
        "dark_ratio": round(dark_ratio, 3),
        "overexposed_ratio": round(overexposed_ratio, 3),
    }

    # 4. Screenshot & Moiré Pattern Heuristics
    exif = pil_img.getexif()
    has_exif = bool(exif)
    screen_ar_match = round(aspect_ratio, 2) in [1.33, 1.78, 2.05, 2.17]

    # 2D FFT Moiré Peak-to-Mean Ratio (PMR)
    f = np.fft.fft2(norm_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    nh, nw = norm_gray.shape
    cy, cx = nh // 2, nw // 2
    cv2.circle(magnitude_spectrum, (cx, cy), 15, 0, -1)
    peak_val = np.max(magnitude_spectrum)
    mean_val = np.mean(magnitude_spectrum)
    pmr = peak_val / mean_val if mean_val > 0 else 0.0

    results["screenshot"] = {
        "has_exif": has_exif,
        "screen_ar_match": screen_ar_match,
        "fft_moire_pmr": round(pmr, 2),
    }

    # 5. OCR Pipeline Detailed Trace & Debug Crops
    ocr_trace = {
        "haar_regions_found": 0,
        "contour_candidates_found": 0,
        "yellow_hsv_candidates_found": 0,
        "raw_ocr_outputs": [],
        "debug_crop_paths": [],
    }

    debug_dir = os.path.join(".", "debug_crops")
    os.makedirs(debug_dir, exist_ok=True)

    # 5a. Haar Cascade
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
    if os.path.exists(cascade_path):
        plate_cascade = cv2.CascadeClassifier(cascade_path)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 10))
        ocr_trace["haar_regions_found"] = len(plates)

    # 5b. Contour Candidates
    y0 = int(h * 0.25)
    roi = bgr[y0:, :]
    rh, rw = roi.shape[:2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c_cnt = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch >= 12 and cw >= 30:
            ar = cw / float(ch)
            if 1.2 <= ar <= 6.5 and cv2.contourArea(cnt) >= 150:
                c_cnt += 1
    ocr_trace["contour_candidates_found"] = c_cnt

    # 5c. Yellow HSV Candidates
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([10, 40, 60]), np.array([45, 255, 255]))
    cnts_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ocr_trace["yellow_hsv_candidates_found"] = len(cnts_yellow)

    # 5d. Raw Tesseract OCR Execution on localized candidate crops
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    crops_to_test = []
    
    # 1. Haar Cascade crop if found
    if ocr_trace["haar_regions_found"] > 0:
        px, py, pw, ph = plates[0]
        c_haar = bgr[max(0, py - 5):min(h, py + ph + 5), max(0, px - 5):min(w, px + pw + 5)]
        if c_haar.size > 0:
            crops_to_test.append((c_haar, "haar_crop"))

    # 2. Top Yellow HSV Mask Crops
    yellow_crops = []
    for cnt in cnts_yellow:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch >= 12 and cw >= 30:
            ar = cw / float(ch)
            if 1.2 <= ar <= 6.5 and cv2.contourArea(cnt) >= 150:
                c_yellow = roi[max(0, y-6):min(rh, y+ch+6), max(0, x-6):min(rw, x+cw+6)]
                if c_yellow.size > 0:
                    yellow_crops.append((c_yellow, abs(ar - 3.2)))

    if yellow_crops:
        yellow_crops.sort(key=lambda t: t[1])
        for idx, (yc, _) in enumerate(yellow_crops[:3]):
            crops_to_test.append((yc, f"yellow_crop_{idx+1}"))

    # 3. Top Edge Contour Crops
    contour_crops = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch >= 12 and cw >= 30:
            ar = cw / float(ch)
            if 1.2 <= ar <= 6.5 and cv2.contourArea(cnt) >= 150:
                c_edge = roi[max(0, y-6):min(rh, y+ch+6), max(0, x-6):min(rw, x+cw+6)]
                if c_edge.size > 0:
                    contour_crops.append((c_edge, abs(ar - 3.2)))

    if contour_crops:
        contour_crops.sort(key=lambda t: t[1])
        for idx, (ec, _) in enumerate(contour_crops[:3]):
            crops_to_test.append((ec, f"contour_crop_{idx+1}"))

    # 4. Fallback Full Lower ROI
    crops_to_test.append((roi, "full_lower_roi"))

    for idx, (crop_img, label) in enumerate(crops_to_test):
        up = cv2.resize(crop_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        b = cv2.GaussianBlur(g, (5, 5), 0)
        t = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9)

        try:
            raw_text = pytesseract.image_to_string(t, config=config).strip()
            cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
            if raw_text or cleaned:
                ocr_trace["raw_ocr_outputs"].append({"crop_label": label, "raw_string": raw_text, "cleaned_string": cleaned})
        except Exception as ocr_err:
            ocr_trace["raw_ocr_outputs"].append({"crop_label": label, "error": str(ocr_err)})

    results["ocr_trace"] = ocr_trace
    return results


def main():
    search_dirs = [
        os.path.expanduser(r"~\Downloads"),
        os.path.join(".", "uploads"),
        ".",
    ]

    target_dir = sys.argv[1] if len(sys.argv) > 1 else None

    image_files = []
    if target_dir and os.path.exists(target_dir):
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            image_files.extend(glob.glob(os.path.join(target_dir, ext)))
    else:
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                    image_files.extend(glob.glob(os.path.join(s_dir, ext)))

    # Deduplicate image files
    image_files = sorted(list(set(image_files)))

    print("=" * 110)
    print(" RAW SIGNAL INSTRUMENTATION TABLE (NO THRESHOLDS / NO VERDICTS APPLIED)")
    print("=" * 110)
    print(
        f"{'FILENAME':<25} | {'RESOLUTION':<11} | {'RAW LAP VAR':<11} | {'NORM 1000px VAR':<15} | {'INTENSITY':<9} | {'FFT PMR':<7} | {'EXIF':<5} | {'OCR RAW STRING'}"
    )
    print("-" * 110)

    for img_path in image_files:
        # Ignore temporary or debug crop images
        if "debug_crop" in img_path or "scratch" in img_path:
            continue
        sig = inspect_image_signals(img_path)
        if "error" in sig:
            continue

        fn = sig["filename"][:24]
        res_str = f"{sig['dimensions']['width']}x{sig['dimensions']['height']}"
        raw_v = sig["blur"]["raw_laplacian_variance"]
        norm_v = sig["blur"]["normalized_1000px_variance"]
        mean_i = sig["brightness"]["mean_intensity"]
        pmr = sig["screenshot"]["fft_moire_pmr"]
        exif_str = "YES" if sig["screenshot"]["has_exif"] else "NO"
        ocr_raws = ", ".join([o.get("cleaned_string", "") for o in sig["ocr_trace"]["raw_ocr_outputs"] if o.get("cleaned_string")]) or "None"

        print(f"{fn:<25} | {res_str:<11} | {raw_v:<11} | {norm_v:<15} | {mean_i:<9} | {pmr:<7} | {exif_str:<5} | {ocr_raws}")

    print("=" * 110)
    print("\n" + "=" * 110)
    print(" STEP 4: DETAILED OCR PIPELINE TRACE BY IMAGE")
    print("=" * 110)

    for img_path in image_files:
        if "debug_crop" in img_path or "scratch" in img_path:
            continue
        sig = inspect_image_signals(img_path)
        if "error" in sig:
            continue

        print(f"\nImage: {sig['filename']}")
        print(f"  - Dimensions: {sig['dimensions']['width']}x{sig['dimensions']['height']} ({sig['dimensions']['megapixels']} MP)")
        print(f"  - Haar Cascade Regions Found: {sig['ocr_trace']['haar_regions_found']}")
        print(f"  - Edge Contour Bounding Candidates: {sig['ocr_trace']['contour_candidates_found']}")
        print(f"  - Yellow HSV Bounding Candidates: {sig['ocr_trace']['yellow_hsv_candidates_found']}")
        print(f"  - Saved Debug Crops: {', '.join(sig['ocr_trace']['debug_crop_paths'])}")
        print("  - Raw OCR Output Before Regex Filtering:")
        for out in sig['ocr_trace']['raw_ocr_outputs']:
            print(f"      [{out.get('crop_label')}]: raw='{out.get('raw_string')}' | cleaned='{out.get('cleaned_string')}'")

    print("\n" + "=" * 110)


if __name__ == "__main__":
    main()
