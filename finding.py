from detector import process_image, match_faces, cosine_similarity
import numpy as np

print("=" * 50)
print("TESTING FACE MATCHING")
print("=" * 50)

# Extract embedding from photo 1
print("\n📸 Processing Photo 1...")
embeddings1 = process_image("C:\\Users\\abhis\\hackathon\\photos\\testimg.jpeg")

# Extract embedding from photo 2
print("\n📸 Processing Photo 2...")
embeddings2 = process_image("C:\\Users\\abhis\\hackathon\\photos\\first.jpeg")

if embeddings1 and embeddings2:
    
    # Check actual similarity score first
    sim = cosine_similarity(
        embeddings1[0]["embedding"],
        embeddings2[0]["embedding"]
    )
    print(f"\n📊 Actual similarity score: {sim:.4f}")
    print("(1.0 = identical, 0.0 = completely different)")
    
    # Try with lower threshold
    stored = [{
        "photo_id": "test-photo-001",
        "s3_url": "https://fake-s3-url.com/photo1.jpg",
        "embedding": embeddings1[0]["embedding"]
    }]
    
    selfie_embedding = embeddings2[0]["embedding"]
    
    print("\n🔍 Testing different thresholds:")
    for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        matches = match_faces(selfie_embedding, stored, threshold=threshold)
        status = "✅ MATCH" if matches else "❌ NO MATCH"
        print(f"Threshold {threshold} → {status}")