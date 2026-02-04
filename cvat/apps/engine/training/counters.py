import redis
from django.conf import settings

r = redis.Redis(
    host=settings.REDIS_INMEM_SETTINGS["HOST"],
    port=settings.REDIS_INMEM_SETTINGS["PORT"],
    db=settings.REDIS_INMEM_SETTINGS["DB"],
)

THRESHOLD = 20

def increment_task_annotation_counter(task_id: int, frames: int = 1) -> int:
    return r.incrby(f"task:{task_id}:auto_frames", frames)

def should_retrain(task_id: int) -> bool:
    return int(r.get(f"task:{task_id}:auto_frames") or 0) >= THRESHOLD

def reset_counter(task_id: int):
    r.delete(f"task:{task_id}:auto_frames")

def training_lock(task_id: int):
    return r.lock(f"task:{task_id}:training", timeout=6 * 60 * 60)
