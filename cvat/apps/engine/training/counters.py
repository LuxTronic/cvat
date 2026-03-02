import redis
from django.conf import settings
import logging
logger = logging.getLogger(__name__)

r = redis.Redis(
    host=settings.REDIS_INMEM_SETTINGS["HOST"],
    port=settings.REDIS_INMEM_SETTINGS["PORT"],
    db=settings.REDIS_INMEM_SETTINGS["DB"],
)

THRESHOLD = 20

def increment(task_id: int, frames: int):
    new_count = r.incrby(f"task:{task_id}:frames", frames)
    logger.info(
        "[COUNTER] Task %s incremented by %d → total=%d",
        task_id,
        frames,
        new_count,
    )
    return new_count


def should_retrain(task_id: int) -> bool:
    count = int(r.get(f"task:{task_id}:frames") or 0)
    decision = count >= THRESHOLD
    logger.info(
        "[COUNTER] Task %s count=%d threshold=%d retrain=%s",
        task_id,
        count,
        THRESHOLD,
        decision,
    )
    return decision


def reset(task_id: int):
    r.delete(f"task:{task_id}:frames")
    logger.info("[COUNTER] Task %s counter reset", task_id)


def training_lock(task_id: int):
    logger.debug("[LOCK] Creating training lock for task %s", task_id)
    return r.lock(f"task:{task_id}:training", timeout=6 * 60 * 60)
