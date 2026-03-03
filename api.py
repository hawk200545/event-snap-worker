from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from urllib.parse import urlparse
import tempfile
import os
import uuid
from dotenv import load_dotenv
from detector import process_image, match_faces
from prisma import Prisma
from bullmq import Worker
from consumer import process_photo_job
import redis

# Load environment variables from .env before using them
load_dotenv()

# Initialize Redis connection safely — fall back to local Redis if REDIS_URL missing
REDIS_URL = os.getenv("REDIS_URL")
if REDIS_URL:
    r = redis.from_url(REDIS_URL)
else:
    print("⚠️ REDIS_URL not set. Falling back to local Redis at redis://localhost:6379")
    r = redis.Redis(host="localhost", port=6379, db=0)

# Initialize Prisma
db = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    print("✅ Connected to NeonDB!")

    parsed = urlparse(REDIS_URL) if REDIS_URL else None
    worker = None
    if parsed:
        redis_opts = {
            "host": parsed.hostname,
            "port": parsed.port or 6379,
            "username": parsed.username or "default",
            "password": parsed.password,
            "ssl": REDIS_URL.startswith("rediss://"),
        }
        worker = Worker("photo-processing", process_photo_job, {"connection": redis_opts})
        print("✅ BullMQ Worker started — listening on 'photo-processing'")

    yield

    if worker:
        await worker.close()
    await db.disconnect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Local uploads folder
UPLOADS_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

def save_file_locally(file_bytes: bytes, filename: str) -> str:
    """Save file locally and return path"""
    ext = os.path.splitext(filename)[1] or ".jpg"
    key = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOADS_DIR, key)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path, key

# ─────────────────────────────────────────
# ROUTE 1 — Health Check
# ─────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "EventSnap ML API is running! 🚀"}

# ─────────────────────────────────────────
# ROUTE 2 — Extract Embedding (called by NestJS faces.service.ts selfieMatch)
# ─────────────────────────────────────────
@app.post("/extract-embedding")
async def extract_embedding(image: UploadFile = File(...)):
    file_bytes = await image.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        embeddings = process_image(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not embeddings:
        raise HTTPException(status_code=422, detail="No face detected in the image")

    return {"embedding": embeddings[0]["embedding"]}

# ─────────────────────────────────────────
# ROUTE 3 — Create Room
# ─────────────────────────────────────────
@app.post("/room/create")
async def create_room(name: str):
    # Create a default organizer user if not exists
    user = await db.user.find_first()
    
    if not user:
        user = await db.user.create(
            data={
                "email": "organizer@eventsnap.com",
                "passwordHash": "hackathon123",
                "fullName": "Event Organizer",
                "role": "ORGANIZER"
            }
        )
    
    room = await db.room.create(
        data={
            "name": name,
            "code": str(uuid.uuid4())[:8].upper(),
            "organizerId": user.id,
            "startsAt": "2026-03-01T00:00:00Z",
            "endsAt": "2026-03-02T00:00:00Z",
        }
    )
    return {
        "success": True,
        "room_id": room.id,
        "room_code": room.code,
        "name": room.name
    }
# ─────────────────────────────────────────
# ROUTE 4 — Upload Photo
# ─────────────────────────────────────────
@app.post("/room/{room_id}/upload")
async def upload_photo(room_id: str, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        # Save file locally
        file_path, storage_key = save_file_locally(file_bytes, file.filename)
        print(f"✅ Saved locally: {file_path}")
        
        # Save to temp file for DeepFace
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        # Extract embeddings
        embeddings = process_image(tmp_path)
        os.unlink(tmp_path)
        
        if not embeddings:
            return {"success": False, "message": "No faces found"}
        
        # Save photo to NeonDB ← REAL DB NOW
        photo = await db.photo.create(
            data={
                "roomId": room_id,
                "storageKey": storage_key,
                "bucket": "local",
                "originalFileName": file.filename,
                "contentType": "image/jpeg",
                "sizeBytes": len(file_bytes),
                "status": "READY"
            }
        )
        print(f"✅ Photo saved to NeonDB: {photo.id}")
        
        # Save embeddings to NeonDB ← REAL DB NOW
        for i, emb in enumerate(embeddings):
            await db.faceembedding.create(
                data={
                    "photoId": photo.id,
                    "roomId": room_id,
                    "embedding": emb["embedding"],
                    "faceIndex": i
                }
            )
        print(f"✅ {len(embeddings)} embeddings saved to NeonDB")
        
        return {
            "success": True,
            "photo_id": photo.id,
            "file_path": file_path,
            "faces_found": len(embeddings)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# ROUTE 5 — Search by Selfie
# ─────────────────────────────────────────
@app.post("/room/{room_id}/search")
async def search_by_selfie(room_id: str, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        # Get selfie embedding
        selfie_embeddings = process_image(tmp_path)
        os.unlink(tmp_path)
        
        if not selfie_embeddings:
            return {"success": False, "message": "No face found in selfie"}
        
        selfie_embedding = selfie_embeddings[0]["embedding"]
        
        # Fetch all embeddings from NeonDB ← REAL DB NOW
        db_embeddings = await db.faceembedding.find_many(
            where={"roomId": room_id},
            include={"photo": True}
        )
        
        if not db_embeddings:
            return {"success": False, "message": "No photos in this room yet"}
        
        # Format for match_faces
        stored_embeddings = [
            {
                "photo_id": emb.photoId,
                "s3_url": emb.photo.storageKey,
                "embedding": emb.embedding
            }
            for emb in db_embeddings
        ]
        
        # Match faces
        matches = match_faces(selfie_embedding, stored_embeddings, threshold=0.75)
        
        return {
            "success": True,
            "matches_found": len(matches),
            "photos": matches
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Guest requests access
@app.post("/room/{room_id}/request")
async def request_access(room_id: str, name: str, phone: str):
    room = await db.room.find_unique(where={"id": room_id})
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    # Create guest session in NeonDB
    guest = await db.guestsession.create(
        data={
            "roomId": room_id,
            "guestToken": str(uuid.uuid4()),
            "displayName": name,
            "status": "ACTIVE"
        }
    )
    
    # Set pending in Redis
    r.set(f"participant:{guest.id}", "pending", ex=3600)
    
    return {
        "success": True,
        "participant_id": guest.id,
        "status": "pending",
        "message": "Waiting for organizer approval"
    }

# Guest polls this every 3 seconds
@app.get("/participant/{participant_id}/status")
async def check_status(participant_id: str):
    status = r.get(f"participant:{participant_id}")
    return {
        "participant_id": participant_id,
        "status": status.decode() if status else "pending"
    }

# Organizer approves
@app.post("/participant/{participant_id}/approve")
async def approve_participant(participant_id: str):
    r.set(f"participant:{participant_id}", "approved", ex=3600)
    return {"success": True, "status": "approved"}

# Organizer rejects
@app.post("/participant/{participant_id}/reject")
async def reject_participant(participant_id: str):
    r.set(f"participant:{participant_id}", "rejected", ex=3600)
    return {"success": True, "status": "rejected"}

# Organizer sees all pending requests
@app.get("/room/{room_id}/requests")
async def get_pending_requests(room_id: str):
    guests = await db.guestsession.find_many(
        where={
            "roomId": room_id,
            "status": "ACTIVE"
        }
    )
    
    pending = []
    for guest in guests:
        status = r.get(f"participant:{guest.id}")
        if not status or status.decode() == "pending":
            pending.append({
                "participant_id": guest.id,
                "name": guest.displayName,
                "joined_at": guest.joinedAt
            })
    
    return {
        "success": True,
        "pending": pending
    }
# ─────────────────────────────────────────
# ROUTE 6 — Room Status
# ─────────────────────────────────────────
@app.get("/room/{room_id}/status")
async def room_status(room_id: str):
    count = await db.faceembedding.count(
        where={"roomId": room_id}
    )
    return {
        "room_id": room_id,
        "exists": count > 0,
        "total_embeddings": count
    }

# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)