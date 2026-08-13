from PIL import Image
import imagehash
from sqlalchemy.orm import Session
from app.models import ImageHash

def analyze_duplicate(image_path: str, db: Session, upload_id: str):
    """
    Uses perceptual hashing (pHash) to find duplicates in the DB.
    
    Why pHash is better than MD5/file hashing for this use case:
    Cryptographic hashes like MD5 change completely if even a single bit of the 
    file changes. If a user uploads an image, then downloads it (which might 
    compress it or strip EXIF metadata), and uploads it again, the MD5 hashes 
    will not match, failing to detect the duplicate.
    
    Perceptual hashing (pHash) looks at the visual features of the image instead 
    of the binary data. Visually similar images will produce similar hashes. 
    By calculating the Hamming distance between two pHashes, we can accurately 
    detect duplicates even if they have been resized, compressed, or slightly altered.
    """
    try:
        img = Image.open(image_path)
        current_hash = str(imagehash.phash(img))
    except Exception as e:
         return {"name": "duplicate", "score": 0.0, "signal": str(e), "confidence": 0.0, "verdict": "unknown"}

    # Fetch all hashes
    all_hashes = db.query(ImageHash).filter(ImageHash.upload_id != upload_id).all()
    
    min_distance = float('inf')
    matched_upload = None
    
    target_hash_obj = imagehash.hex_to_hash(current_hash)
    
    for row in all_hashes:
        db_hash_obj = imagehash.hex_to_hash(row.phash)
        distance = target_hash_obj - db_hash_obj
        if distance < min_distance:
            min_distance = distance
            matched_upload = row.upload_id
            
    # Threshold for phash is typically <= 5 for duplicates
    threshold = 5
    if min_distance <= threshold:
        return {
            "name": "duplicate", 
            "score": round(max(0, 1.0 - (min_distance / 64.0)), 2), 
            "signal": f"hash_distance={min_distance}", 
            "match_id": matched_upload, 
            "confidence": 0.95, 
            "verdict": "rejected", 
            "current_hash": current_hash
        }
    
    return {
        "name": "duplicate", 
        "score": 1.0, 
        "signal": "unique", 
        "confidence": 0.95, 
        "verdict": "clean", 
        "current_hash": current_hash
    }
