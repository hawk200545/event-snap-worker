import cv2
import os
from deepface import DeepFace

# ─────────────────────────────────────────
# STEP 1 — Validate Image
# ─────────────────────────────────────────
def validate_image(image_path: str):
    """
    Check if image is good enough to process
    """
    img = cv2.imread(image_path)
    
    if img is None:
        return False, "Could not load image"
    
    h, w = img.shape[:2]
    if h < 50 or w < 50:
        return False, "Image too small"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < 50:
        return False, f"Image too blurry (score: {blur_score:.1f})"
    
    print(f"✅ Image valid — size: {w}x{h}, blur score: {blur_score:.1f}")
    return True, "OK"


# ─────────────────────────────────────────
# STEP 3 — Extract Face Embeddings
# ─────────────────────────────────────────
def extract_embeddings(image_path: str):
    """
    Detect all faces in image
    Return list of embeddings
    """
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False,
        )

        embeddings = []
        for r in results:
            # Skip low-confidence detections (retinaface provides this)
            if r.get("face_confidence", 1.0) < 0.85:
                continue
            embeddings.append({
                "embedding": r["embedding"],
                "facial_area": r["facial_area"]
            })
        
        print(f"👤 Found {len(embeddings)} face(s)")
        return embeddings
    
    except Exception as e:
        print(f"❌ Error extracting embeddings: {e}")
        return []


# ─────────────────────────────────────────
# STEP 3b — Crop Face Thumbnail
# ─────────────────────────────────────────
def crop_face_thumbnail(image_path: str, facial_area: dict, size: int = 160) -> bytes:
    """Crop the face region from image, resize to square, return JPEG bytes."""
    img = cv2.imread(image_path)
    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    face_crop = img[y:y2, x:x2]
    face_crop = cv2.resize(face_crop, (size, size))
    _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ─────────────────────────────────────────
# STEP 4 — Full Pipeline
# ─────────────────────────────────────────
def process_image(image_path: str):
    """
    Run full pipeline on an uploaded photo
    Returns embeddings if successful
    """
    print(f"\n🔄 Processing: {image_path}")

    valid, message = validate_image(image_path)
    if not valid:
        print(f"❌ Validation failed: {message}")
        return []

    # Run detection on the original file so facial_area coords
    # stay consistent with the file consumer.py uses for cropping
    return extract_embeddings(image_path)


# ─────────────────────────────────────────
# TEST — Run this to verify everything works
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    # Test with your image
    test_image = "C:\\Users\\abhis\\hackathon\\photos\\testimg.jpeg"
    
    print("=" * 50)
    print("TESTING DETECTOR PIPELINE")
    print("=" * 50)
    
    embeddings = process_image(test_image)
    
    if embeddings:
        print(f"\n✅ SUCCESS!")
        print(f"Number of faces found: {len(embeddings)}")
        print(f"Embedding size: {len(embeddings[0]['embedding'])}")
        print(f"Facial area: {embeddings[0]['facial_area']}")
    else:
        print("\n❌ No faces found or error occurred")