from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
from detector import process_image, match_faces, cosine_similarity

app = FastAPI()

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────
# Temporary in-memory storage
# (will replace with PostgreSQL later)
# ─────────────────────────────────────────
rooms = {}  # room_id → list of embeddings


# ─────────────────────────────────────────
# ROUTE 1 — Health Check
# ─────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "EventSnap ML API is running! 🚀"}


# ─────────────────────────────────────────
# ROUTE 2 — Upload Photo to Room
# ─────────────────────────────────────────
@app.post("/room/{room_id}/upload")
async def upload_photo(room_id: str, file: UploadFile = File(...)):
    """
    Guest uploads a photo to the room
    Extract faces and store embeddings
    """
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Save to temp file for DeepFace
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        # Process image
        embeddings = process_image(tmp_path)
        
        # Cleanup temp file
        os.unlink(tmp_path)
        
        if not embeddings:
            return {
                "success": False,
                "message": "No faces found in photo"
            }
        
        # Store embeddings in memory (keyed by room)
        if room_id not in rooms:
            rooms[room_id] = []
        
        photo_id = f"{room_id}_{file.filename}"
        
        for emb in embeddings:
            rooms[room_id].append({
                "photo_id": photo_id,
                "s3_url": f"placeholder/{photo_id}",  # will be real S3 URL later
                "embedding": emb["embedding"]
            })
        
        print(f"Room {room_id} now has {len(rooms[room_id])} embeddings")
        
        return {
            "success": True,
            "photo_id": photo_id,
            "faces_found": len(embeddings),
            "message": f"Successfully processed {len(embeddings)} face(s)"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# ROUTE 3 — Search by Selfie
# ─────────────────────────────────────────
@app.post("/room/{room_id}/search")
async def search_by_selfie(room_id: str, file: UploadFile = File(...)):
    """
    Guest uploads selfie
    Returns all matching photos from the room
    """
    try:
        # Check room exists
        if room_id not in rooms or not rooms[room_id]:
            return {
                "success": False,
                "message": "Room not found or no photos uploaded yet"
            }
        
        # Read selfie bytes
        file_bytes = await file.read()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        # Get selfie embedding
        selfie_embeddings = process_image(tmp_path)
        os.unlink(tmp_path)
        
        if not selfie_embeddings:
            return {
                "success": False,
                "message": "No face found in selfie"
            }
        
        selfie_embedding = selfie_embeddings[0]["embedding"]
        
        # Match against room embeddings
        matches = match_faces(
            selfie_embedding,
            rooms[room_id],
            threshold=0.75
        )
        
        return {
            "success": True,
            "matches_found": len(matches),
            "photos": matches
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# ROUTE 4 — Check Room Status
# ─────────────────────────────────────────
@app.get("/room/{room_id}/status")
def room_status(room_id: str):
    """
    Check how many photos are in a room
    """
    if room_id not in rooms:
        return {
            "room_id": room_id,
            "exists": False,
            "total_embeddings": 0
        }
    
    return {
        "room_id": room_id,
        "exists": True,
        "total_embeddings": len(rooms[room_id])
    }


# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)