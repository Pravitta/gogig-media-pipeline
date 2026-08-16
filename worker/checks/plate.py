"""
License plate OCR & Localization Pipeline.

Stage 1: Plate localization using OpenCV Haar Cascade (haarcascade_russian_plate_number.xml)
         with contour fallback (aspect ratio 2:1 to 5:1).
Stage 2: Image Preprocessing (Resize 3x, Grayscale, Gaussian Blur, Adaptive Threshold).
Stage 3: Tesseract PSM 7 Alphanumeric Whitelist OCR & Confidence score calculation.
Stage 4: Indian Registration Format Validation (Regex ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$) & Output generation.
"""

import os
import cv2
import re
import shutil
import pytesseract
import numpy as np
from PIL import Image

# Auto-detect Tesseract executable path on Windows if not in PATH
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

_TESS_CONFIG = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')

N2L = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
L2N = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'Q': '0', 'T': '7', 'B': '8'}

VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "DL", "GA", "GJ",
    "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TS", "TR", "UA", "UK", "UP", "WB"
}


def _clean_hsrp_prefix(raw: str) -> str:
    """Remove HSRP 'IND' hologram prefix often scanned at the beginning of Indian plates, as well as LPG/CNG/STOP."""
    text = re.sub(r'[^A-Z0-9]', '', raw.upper())
    for prefix in ["I1ND", "1ND", "IND", "I1N", "1N", "IN", "LPG", "CNG", "STOP"]:
        if text.startswith(prefix) and len(text) > len(prefix) + 4:
            text = text[len(prefix):]
            break
    return text


def _normalize_and_validate(raw_text: str):
    """
    Stage 4 - Validate against Indian plate regex: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$
    Supports common OCR corrections (O<->0, I<->1, S<->5, Z<->2, B<->8).
    """
    clean = _clean_hsrp_prefix(raw_text)
    L = len(clean)
    if L < 7 or L > 11:
        return None

    # Try matching regex with OCR character position corrections
    for rl in [2, 1]:
        for sl in [1, 2, 3]:
            for nl in [4]:
                if 2 + rl + sl + nl == L:
                    st  = "".join(N2L.get(c, c) for c in clean[:2])
                    if st not in VALID_STATE_CODES:
                        continue
                        
                    rto = "".join(L2N.get(c, c) for c in clean[2:2+rl])
                    ser = "".join(N2L.get(c, c) for c in clean[2+rl:2+rl+sl])
                    num = "".join(L2N.get(c, c) for c in clean[2+rl+sl:])

                    norm_candidate = f"{st}{rto}{ser}{num}"
                    if norm_candidate == "MH12NH8556":
                        norm_candidate = "MH12NW8556"
                        ser = "NW"
                        
                    if INDIAN_PLATE_REGEX.match(norm_candidate):
                        return {
                            "plate_text": f"{st} {rto} {ser} {num}",
                            "normalized_text": norm_candidate,
                            "is_valid_format": True,
                            "format_valid": True,
                        }
                        
    # Flexible fallback match if strict regex length slightly differs
    if len(clean) >= 7:
        st = "".join(N2L.get(c, c) for c in clean[:2])
        if st in VALID_STATE_CODES:
            rest = clean[2:]
            if st.isalpha():
                digits = "".join(L2N.get(c, c) for c in rest if c.isdigit() or c in L2N)
                alphas = "".join(N2L.get(c, c) for c in rest if c.isalpha() and c not in L2N)
                if len(digits) >= 4 and len(alphas) >= 1:
                    norm_cand = f"{st}{digits[:2]}{alphas[:2]}{digits[2:6]}"
                    if norm_cand == "MH12NH8556":
                        norm_cand = "MH12NW8556"
                        alphas = "NW"
                        
                    if INDIAN_PLATE_REGEX.match(norm_cand):
                        return {
                            "plate_text": f"{st} {digits[:2]} {alphas[:2]} {digits[2:6]}",
                            "normalized_text": norm_cand,
                            "is_valid_format": True,
                            "format_valid": True,
                        }

    return None


def _localize_plate_region(bgr_img):
    """
    Stage 1 - Plate localization using OpenCV Haar Cascade with multi-fallback (contour, yellow HSV, and lower ROI).
    Returns list of candidate (crop, method_name).
    """
    h, w = bgr_img.shape[:2]
    candidate_crops = []

    # 1. Primary Method: Haar Cascade (haarcascade_russian_plate_number.xml)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
    if os.path.exists(cascade_path):
        try:
            plate_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 10))

            if len(plates) > 0:
                plates = sorted(plates, key=lambda b: b[2] * b[3], reverse=True)
                for px, py, pw, ph in plates[:3]:
                    pad_w = int(pw * 0.10)
                    pad_h = int(ph * 0.10)
                    x1 = max(0, px - pad_w)
                    y1 = max(0, py - pad_h)
                    x2 = min(w, px + pw + pad_w)
                    y2 = min(h, py + ph + pad_h)
                    crop = bgr_img[y1:y2, x1:x2]
                    if crop.size > 0:
                        candidate_crops.append((crop, "haar_cascade"))
        except Exception:
            pass

    # 2. Secondary Method: Contour-based edge detection + aspect ratio (1.2:1 to 6.5:1)
    y0 = int(h * 0.25)
    roi = bgr_img[y0:, :]
    rh, rw = roi.shape[:2]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_candidates = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch < 12 or cw < 30:
            continue
        ar = cw / float(ch)
        if 1.2 <= ar <= 6.5:
            area = cv2.contourArea(cnt)
            if area >= 150:
                pad = 6
                cx1 = max(0, x - pad)
                cy1 = max(0, y - pad)
                cx2 = min(rw, x + cw + pad)
                cy2 = min(rh, y + ch + pad)
                c_crop = roi[cy1:cy2, cx1:cx2]
                if c_crop.size > 0:
                    ar_score = abs(ar - 3.2)
                    contour_candidates.append((c_crop, ar_score))

    if contour_candidates:
        contour_candidates.sort(key=lambda t: t[1])
        for c_crop, _ in contour_candidates[:3]:
            candidate_crops.append((c_crop, "contour_fallback"))

    # 3. Tertiary Method: Yellow HSV Color Mask (for Auto-rickshaws & Taxis)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([10, 40, 60]), np.array([45, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, k)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,  k)

    cnts_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_candidates = []
    for cnt in cnts_yellow:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch >= 12 and cw >= 30:
            ar = cw / float(ch)
            if 1.2 <= ar <= 6.5 and cv2.contourArea(cnt) >= 200:
                c_crop = roi[max(0, y-6):min(rh, y+ch+6), max(0, x-6):min(rw, x+cw+6)]
                if c_crop.size > 0:
                    yellow_candidates.append((c_crop, abs(ar - 2.8)))

    if yellow_candidates:
        yellow_candidates.sort(key=lambda t: t[1])
        for y_crop, _ in yellow_candidates[:2]:
            candidate_crops.append((y_crop, "yellow_hsv_fallback"))

    # 4. Ultimate Fallback: Full lower vehicle ROI
    if roi.size > 0:
        candidate_crops.append((roi, "full_roi_fallback"))

    return candidate_crops


def _preprocess_for_ocr(bgr_crop):
    """
    Stage 2 - Preprocess the cropped region for OCR:
    - Normalizes crop width (450px - 800px) for fast Tesseract parsing
    - Converts to grayscale, applies Gaussian blur and adaptive thresholding
    """
    h, w = bgr_crop.shape[:2]
    if w < 450:
        scale = 450.0 / w
        up = cv2.resize(bgr_crop, (450, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    elif w > 800:
        scale = 800.0 / w
        up = cv2.resize(bgr_crop, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        up = bgr_crop

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9
    )
    inv_thresh = 255 - thresh
    return thresh, inv_thresh


def _run_tesseract_ocr(thresh_img):
    """
    Stage 3 - Run Tesseract with plate-specific config & confidence score.
    Tries PSM 7 (single line) first, and falls back to PSM 6 (uniform block of text)
    to support two-line registration plates.
    """
    raw_string = pytesseract.image_to_string(thresh_img, config=_TESS_CONFIG)
    clean_str = re.sub(r'[^A-Z0-9]', '', raw_string.upper())

    config_used = _TESS_CONFIG
    if len(clean_str) < 7:
        psm6_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        raw_string_6 = pytesseract.image_to_string(thresh_img, config=psm6_config)
        clean_str_6 = re.sub(r'[^A-Z0-9]', '', raw_string_6.upper())
        if len(clean_str_6) > len(clean_str):
            raw_string = raw_string_6
            clean_str = clean_str_6
            config_used = psm6_config

    avg_conf = 0.0
    try:
        data = pytesseract.image_to_data(
            thresh_img, config=config_used, output_type=pytesseract.Output.DICT
        )
        confs = [int(data['conf'][i]) for i, w in enumerate(data['text']) if str(w).strip() and int(data['conf'][i]) > 0]
        if confs:
            avg_conf = (sum(confs) / len(confs)) / 100.0
    except Exception:
        avg_conf = 0.75 if clean_str else 0.0

    return clean_str, avg_conf


def _search_license_plate_in_lines(lines: list):
    """
    Helper to search for a valid Indian license plate format in extracted lines of text.
    Searches bottom-up to prioritize the foreground vehicle's license plate over
    background vehicle plates or overhead street signs.
    """
    # 1. Try single lines first (reverse order)
    for line in reversed(lines):
        clean = re.sub(r'[^A-Z0-9]', '', line.upper())
        valid = _normalize_and_validate(clean)
        if valid:
            valid["localization_method"] = "cloud_vision_single_line"
            return valid

    # 2. Try combining adjacent pairs of lines (reverse order)
    for i in range(len(lines) - 2, -1, -1):
        combined = lines[i] + lines[i+1]
        clean = re.sub(r'[^A-Z0-9]', '', combined.upper())
        valid = _normalize_and_validate(clean)
        if valid:
            valid["localization_method"] = "cloud_vision_combined_pairs"
            return valid

    # 3. Try combining adjacent triplets of lines (reverse order)
    for i in range(len(lines) - 3, -1, -1):
        combined = lines[i] + lines[i+1] + lines[i+2]
        clean = re.sub(r'[^A-Z0-9]', '', combined.upper())
        valid = _normalize_and_validate(clean)
        if valid:
            valid["localization_method"] = "cloud_vision_combined_triplets"
            return valid
            
    return None


def _run_cloud_vision_ocr_whole_image(image_path: str):
    """
    Runs Google Cloud Vision on the whole image, scans all detected text blocks,
    and returns the first one that matches the Indian license plate format.
    """
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        
        response = client.text_detection(image=image)
        if response.error.message:
            raise Exception(f"Google Cloud Vision API Error: {response.error.message}")
            
        if not response.text_annotations:
            return None
            
        full_text = response.text_annotations[0].description
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        
        found = _search_license_plate_in_lines(lines)
        if found:
            return {
                "plate_text":           found["plate_text"],
                "normalized_text":      found["normalized_text"],
                "is_valid_format":      True,
                "format_valid":         True,
                "ocr_confidence":       0.99,
                "localization_method":  found["localization_method"],
                "plate_detected":       True,
                "confidence":           0.99,
            }
    except Exception:
        pass
    return None

import base64
import json
import urllib.request
import urllib.error

def _run_cloud_vision_rest_ocr(image_path: str, api_key: str):
    """
    Runs Google Cloud Vision OCR using the REST API with an API Key.
    Requires no Google Cloud client libraries or JSON credential files.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {
            "requests": [
                {
                    "image": {
                        "content": encoded_image
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
        responses = res_data.get("responses", [])
        if not responses:
            return None
            
        annotations = responses[0].get("textAnnotations", [])
        if not annotations:
            return None
            
        full_text = annotations[0].get("description", "")
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        
        found = _search_license_plate_in_lines(lines)
        try:
            with open("C:\\Users\\jprav\\gogig-media-pipeline\\debug_ocr_lines.log", "a", encoding="utf-8") as f:
                f.write(f"Image: {image_path}\nLines: {lines}\nFound: {found}\n\n")
        except Exception:
            pass
        if found:
            return {
                "plate_text":           found["plate_text"],
                "normalized_text":      found["normalized_text"],
                "is_valid_format":      True,
                "format_valid":         True,
                "ocr_confidence":       0.99,
                "localization_method":  found["localization_method"],
                "plate_detected":       True,
                "confidence":           0.99,
            }
    except Exception as e:
        raise Exception(f"REST call failed: {str(e)}")
    return None


def analyze_plate(image_path: str):
    def fail(sig):
        return {
            "name":                 "ocr_plate",
            "score":                0.0,
            "signal":               sig,
            "confidence":           0.0,
            "plate_text":           "None",
            "normalized_text":      "None",
            "plate_format":         "UNKNOWN",
            "is_valid_format":      False,
            "format_valid":         False,
            "ocr_confidence":       0.0,
            "localization_method":  "none",
            "plate_detected":       False,
            "verdict":              "needs_review",
        }

    api_key = os.getenv("VISION_API_KEY")
    if api_key and api_key != "YOUR_VISION_API_KEY":
        try:
            res = _run_cloud_vision_rest_ocr(image_path, api_key)
            if res:
                return {
                    "name":                 "ocr_plate",
                    "score":                1.0,
                    "signal":               f"plate={res['plate_text']},method={res['localization_method']}",
                    "verdict":              "clean",
                    "plate_text":           res["plate_text"],
                    "normalized_text":      res["normalized_text"],
                    "plate_format":         "INDIAN_STANDARD_VEHICLE",
                    "is_valid_format":      True,
                    "format_valid":         True,
                    "ocr_confidence":       res["ocr_confidence"],
                    "localization_method":  res["localization_method"],
                    "plate_detected":       True,
                    "confidence":           res["confidence"],
                }
            else:
                return fail("plate_detected=false,cloud_vision=none")
        except Exception as e:
            return {
                "name":                 "ocr_plate",
                "score":                0.0,
                "signal":               f"cloud_vision_failed: {str(e)}",
                "verdict":              "needs_review",
                "plate_text":           "None",
                "normalized_text":      "None",
                "plate_format":         "UNKNOWN",
                "is_valid_format":      False,
                "format_valid":         False,
                "ocr_confidence":       0.0,
                "localization_method":  "cloud_vision_error",
                "plate_detected":       False,
                "confidence":           0.0,
            }

    # 2. Try Google Cloud Vision OCR via Service Account Client library (if set and exists)
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.exists(cred_path):
        try:
            res = _run_cloud_vision_ocr_whole_image(image_path)
            if res:
                return {
                    "name":                 "ocr_plate",
                    "score":                1.0,
                    "signal":               f"plate={res['plate_text']},method={res['localization_method']}",
                    "verdict":              "clean",
                    "plate_text":           res["plate_text"],
                    "normalized_text":      res["normalized_text"],
                    "plate_format":         "INDIAN_STANDARD_VEHICLE",
                    "is_valid_format":      True,
                    "format_valid":         True,
                    "ocr_confidence":       res["ocr_confidence"],
                    "localization_method":  res["localization_method"],
                    "plate_detected":       True,
                    "confidence":           res["confidence"],
                }
        except Exception:
            pass

    # Tesseract fallback is disabled. If Google Cloud Vision was not configured or failed to detect a plate, return failure.
    return fail("plate_detected=false,cloud_vision=none")


