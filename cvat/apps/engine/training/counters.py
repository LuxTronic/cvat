import logging

import redis
from django.conf import settings

logger = logging.getLogger(__name__)

r = redis.Redis(
    host=settings.REDIS_INMEM_SETTINGS["HOST"],
    port=settings.REDIS_INMEM_SETTINGS["PORT"],
    db=settings.REDIS_INMEM_SETTINGS["DB"],
)

THRESHOLD = 20


def _frames_key(task_id: int, mode: str) -> str:
    return f"task:{task_id}:{mode}:frames"


def _lock_key(task_id: int, mode: str) -> str:
    return f"task:{task_id}:{mode}:training"


def increment(task_id: int, frames: int, mode: str = "detection"):
    new_count = r.incrby(_frames_key(task_id, mode), frames)
    logger.info(
        "[COUNTER] Task %s mode=%s incremented by %d -> total=%d",
        task_id,
        mode,
        frames,
        new_count,
    )
    return new_count


def should_retrain(task_id: int, mode: str = "detection") -> bool:
    count = int(r.get(_frames_key(task_id, mode)) or 0)
    decision = count >= THRESHOLD
    logger.info(
        "[COUNTER] Task %s mode=%s count=%d threshold=%d retrain=%s",
        task_id,
        mode,
        count,
        THRESHOLD,
        decision,
    )
    return decision


def reset(task_id: int, mode: str = "detection"):
    r.delete(_frames_key(task_id, mode), _lock_key(task_id, mode))
    logger.info("[COUNTER] Task %s mode=%s counter and training lock reset", task_id, mode)


def training_lock(task_id: int, mode: str = "detection"):
    logger.debug("[LOCK] Creating training lock for task %s mode=%s", task_id, mode)
    return r.lock(_lock_key(task_id, mode), timeout=6 * 60 * 60)
