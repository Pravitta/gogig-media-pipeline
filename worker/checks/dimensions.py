"""
Image Dimension & Aspect Ratio Validation check.
Measures image resolution, total megapixels, and aspect ratio flags.
"""

from PIL import Image

def analyze_dimensions(image_path: str):
    try:
        img = Image.open(image_path)
        w, h = img.size
        megapixels = (w * h) / 1000000.0
        aspect_ratio = max(w, h) / float(min(w, h)) if min(w, h) > 0 else 1.0

        signals = [f"res={w}x{h}", f"mp={megapixels:.2f}MP", f"ar={aspect_ratio:.2f}"]
        verdict = "clean"
        score = 1.0

        # Low resolution check (< 0.3 MP or < 480px short side)
        if min(w, h) < 400 or megapixels < 0.25:
            verdict = "needs_review"
            score = 0.5
            signals.append("low_resolution")

        # Extremely tall or wide strip check (unusual crop)
        if aspect_ratio > 3.0:
            verdict = "needs_review"
            score = min(score, 0.6)
            signals.append("extreme_aspect_ratio")

        return {
            "name": "dimensions",
            "score": round(score, 2),
            "signal": ",".join(signals),
            "verdict": verdict,
            "width": w,
            "height": h,
            "megapixels": round(megapixels, 2),
            "confidence": 0.95
        }
    except Exception as e:
        return {
            "name": "dimensions",
            "score": 0.0,
            "signal": f"error:{str(e)}",
            "verdict": "unknown",
            "confidence": 0.0
        }
