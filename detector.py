import cv2
import numpy as np
import os
import tempfile
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
# STEP 2 — Preprocess Image
# ─────────────────────────────────────────
def preprocess_image(image_path: str):
    """
    Resize large images to save RAM on t2.micro
    """
    img = cv2.imread(image_path)
    
    max_size = 800
    h, w = img.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        print(f"📐 Resized from {w}x{h} to {new_w}x{new_h}")
    else:
        print(f"📐 No resize needed — {w}x{h}")
    
    # Save processed image to temp file
    processed_path = image_path.replace(".jpg", "_processed.jpg").replace(".jpeg", "_processed.jpeg").replace(".png", "_processed.png")
    cv2.imwrite(processed_path, img)
    
    return processed_path


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
        model_name="Facenet",
        detector_backend="retinaface",  # keep best detector
        enforce_detection=False
)
        
        embeddings = []
        for r in results:
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
# STEP 4 — Match Faces
# ─────────────────────────────────────────
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_faces(selfie_embedding: list, stored_embeddings: list, threshold=0.68):
    """
    Compare selfie against all stored embeddings
    Return matched photos
    """
    matches = []
    
    for item in stored_embeddings:
        similarity = cosine_similarity(selfie_embedding, item["embedding"])
        
        if similarity >= threshold:
            matches.append({
                "photo_id": item["photo_id"],
                "s3_url": item["s3_url"],
                "similarity": round(float(similarity), 3)
            })
    
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    print(f"✅ Found {len(matches)} matching photos")
    return matches


# ─────────────────────────────────────────
# STEP 5 — Full Pipeline
# ─────────────────────────────────────────
def process_image(image_path: str):
    """
    Run full pipeline on an uploaded photo
    Returns embeddings if successful
    """
    print(f"\n🔄 Processing: {image_path}")
    
    # Validate
    valid, message = validate_image(image_path)
    if not valid:
        print(f"❌ Validation failed: {message}")
        return []
    
    # Preprocess
    processed_path = preprocess_image(image_path)
    
    # Extract embeddings
    embeddings = extract_embeddings(processed_path)
    
    # Cleanup processed file
    if processed_path != image_path and os.path.exists(processed_path):
        os.remove(processed_path)
    
    return embeddings


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