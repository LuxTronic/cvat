import logging
import django_rq
from django.conf import settings

from cvat.apps.engine.training.counters import (
    increment,
    training_lock,
)
from cvat.apps.engine.training.retrain_worker import retrain_task_model

logger = logging.getLogger(__name__)

THRESHOLD = 20  # number of newly annotated frames required to retrain


def on_annotation_saved(*, task_id: int, frames: int):
    """
    Called whenever new annotated frames are added to a task.

    task_id : Task that received new supervision
    frames  : Number of newly annotated frames
    """
    logger.info(
        "[TRAINING-TRIGGER] on_new_annotations(task_id=%s, frames=%s)",
        task_id,
        frames,
    )

    # Increment Redis counter
    count = increment(task_id, frames)

    logger.info(
        "[TRAINING-TRIGGER] Task %s supervision count = %s / %s",
        task_id,
        count,
        THRESHOLD,
    )

    if count < THRESHOLD:
        logger.info(
            "[TRAINING-TRIGGER] Task %s below threshold, not retraining yet",
            task_id,
        )
        return

    # Prevent concurrent retraining
    lock = training_lock(task_id)
    if not lock.acquire(blocking=False):
        logger.warning(
            "[TRAINING-TRIGGER] Task %s retraining already in progress, skipping",
            task_id,
        )
        return

    logger.info(
        "[TRAINING-TRIGGER] Task %s threshold reached, enqueuing retraining",
        task_id,
    )

    # Enqueue background retraining job
    queue = django_rq.get_queue(settings.CVAT_QUEUES.IMPORT_DATA.value)
    queue.enqueue(retrain_task_model, task_id)
