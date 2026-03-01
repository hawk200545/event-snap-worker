import json
from prisma import Prisma
from dotenv import load_dotenv

load_dotenv()
db = Prisma()

async def connect():
    await db.connect()
    print("✅ Connected to NeonDB!")

async def disconnect():
    await db.disconnect()

# ─────────────────────────────────────────
# ROOM operations
# ─────────────────────────────────────────
async def create_room(name: str, organizer_id: str):
    import uuid
    room = await db.room.create(
        data={
            "name": name,
            "code": str(uuid.uuid4())[:8].upper(),
            "organizerId": organizer_id,
            "startsAt": "2026-03-01T00:00:00Z",
            "endsAt": "2026-03-02T00:00:00Z",
        }
    )
    return room

async def get_room(room_id: str):
    room = await db.room.find_unique(
        where={"id": room_id}
    )
    return room

# ─────────────────────────────────────────
# PHOTO operations
# ─────────────────────────────────────────
async def save_photo(room_id: str, s3_url: str, filename: str, size: int):
    import uuid
    photo = await db.photo.create(
        data={
            "roomId": room_id,
            "storageKey": str(uuid.uuid4()),
            "bucket": "eventsnap-photos",
            "originalFileName": filename,
            "contentType": "image/jpeg",
            "sizeBytes": size,
            "status": "UPLOADED"
        }
    )
    return photo

# ─────────────────────────────────────────
# EMBEDDING operations
# ─────────────────────────────────────────
async def save_embedding(photo_id: str, room_id: str, embedding: list, face_index: int = 0):
    emb = await db.faceembedding.create(
        data={
            "photoId": photo_id,
            "roomId": room_id,
            "embedding": embedding,
            "faceIndex": face_index
        }
    )
    return emb

async def get_embeddings_by_room(room_id: str):
    embeddings = await db.faceembedding.find_many(
        where={"roomId": room_id},
        include={"photo": True}
    )
    
    result = []
    for emb in embeddings:
        result.append({
            "photo_id": emb.photoId,
            "s3_url": emb.photo.storageKey,
            "embedding": emb.embedding
        })
    
    return result

# ─────────────────────────────────────────
# GUEST SESSION operations
# ─────────────────────────────────────────
async def create_guest_session(room_id: str, display_name: str):
    import uuid
    guest = await db.guestsession.create(
        data={
            "roomId": room_id,
            "guestToken": str(uuid.uuid4()),
            "displayName": display_name,
            "status": "ACTIVE"
        }
    )
    return guest

async def get_guest_session(guest_token: str):
    guest = await db.guestsession.find_unique(
        where={"guestToken": guest_token}
    )
    return guest