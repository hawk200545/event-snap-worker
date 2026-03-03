import os
import tempfile
import uuid
from datetime import datetime, timezone

import boto3
import cv2
from dotenv import load_dotenv

load_dotenv()

from prisma import Prisma
from detector import process_image, crop_face_thumbnail

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

BUCKET = os.getenv("AWS_S3_BUCKET", "eventsnap-dev")


async def process_photo_job(job, token):
    """BullMQ job handler for 'process-photo' jobs."""
    data = job.data  # { photoId, roomId, storageKey }
    photo_id = data["photoId"]
    room_id = data["roomId"]
    storage_key = data["storageKey"]

    db = Prisma()
    await db.connect()

    try:
        # 1. Download photo from S3 once — reuse for thumbnail + face crops
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            s3.download_fileobj(BUCKET, storage_key, tmp)
            tmp_path = tmp.name

        # 2. Generate photo thumbnail (resize to max 800px) → upload to S3
        thumb_key = f"rooms/{room_id}/thumbs/thumb_{photo_id}.jpg"
        img = cv2.imread(tmp_path)
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        s3.put_object(Bucket=BUCKET, Key=thumb_key, Body=buf.tobytes(), ContentType="image/jpeg")

        # 3. Extract face embeddings
        embeddings = process_image(tmp_path)

        # 4. For each face: crop thumbnail → upload → create FaceEmbedding record
        for i, emb in enumerate(embeddings):
            face_bytes = crop_face_thumbnail(tmp_path, emb["facial_area"])

            face_thumb_key = f"rooms/{room_id}/thumbs/face_{photo_id}_{i}.jpg"
            s3.put_object(Bucket=BUCKET, Key=face_thumb_key, Body=face_bytes, ContentType="image/jpeg")

            vec = "[" + ",".join(str(x) for x in emb["embedding"]) + "]"
            await db.execute_raw(
                '''INSERT INTO "FaceEmbedding" (id, "photoId", "roomId", "faceIndex", embedding, "faceThumbKey", "createdAt")
                   VALUES ($1, $2, $3, $4, $5::vector, $6, NOW())''',
                str(uuid.uuid4()), photo_id, room_id, i, vec, face_thumb_key,
            )

        os.unlink(tmp_path)

        # 5. Update Photo → READY
        await db.photo.update(
            where={"id": photo_id},
            data={
                "status": "READY",
                "thumbnailKey": thumb_key,
                "processedAt": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"✅ Photo {photo_id} processed — {len(embeddings)} face(s)")

    except Exception as e:
        print(f"❌ Failed to process photo {photo_id}: {e}")
        await db.photo.update(
            where={"id": photo_id},
            data={"status": "FAILED", "processingError": str(e)},
        )
        raise  # Let BullMQ mark job as failed for retry

    finally:
        await db.disconnect()
