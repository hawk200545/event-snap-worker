"""One-time script: reset FAILED photos to PROCESSING and re-enqueue them."""
import asyncio
import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from bullmq import Queue
from prisma import Prisma

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


async def main():
    db = Prisma()
    await db.connect()

    failed = await db.photo.find_many(where={"status": "FAILED"})
    print(f"Found {len(failed)} FAILED photo(s)")

    if not failed:
        await db.disconnect()
        return

    parsed = urlparse(REDIS_URL)
    queue = Queue("photo-processing", {
        "connection": {
            "host": parsed.hostname,
            "port": parsed.port or 6379,
            "username": parsed.username or "default",
            "password": parsed.password,
            "ssl": REDIS_URL.startswith("rediss://"),
        }
    })

    for photo in failed:
        await db.photo.update(
            where={"id": photo.id},
            data={"status": "PROCESSING", "processingError": None},
        )
        await queue.add("process-photo", {
            "photoId": photo.id,
            "roomId": photo.roomId,
            "storageKey": photo.storageKey,
        }, {
            "jobId": f"photo:{photo.id}:reprocess",
            "attempts": 3,
            "backoff": {"type": "exponential", "delay": 2000},
        })
        print(f"Re-enqueued: {photo.id} ({photo.originalFileName})")

    await queue.close()
    await db.disconnect()
    print("Done.")


asyncio.run(main())
