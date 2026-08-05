import logging

import django_rq
from django.conf import settings

from cvat.apps.engine.training.counters import increment, training_lock
from cvat.apps.engine.training.retrain_worker import (
    retrain_classification_task_model,
    retrain_detection_task_model,
)

logger = logging.getLogger(__name__)

THRESHOLD = 20  # number of newly annotated frames required to retrain


def _trigger_mode(*, task_id: int, frames: int, mode: str):
    if frames <= 0:
        return

    logger.info(
        "[TRAINING-TRIGGER] on_new_annotations(task_id=%s, mode=%s, frames=%s)",
        task_id,
        mode,
        frames,
    )

    count = increment(task_id, frames, mode=mode)
    logger.info(
        "[TRAINING-TRIGGER] Task %s mode=%s supervision count = %s / %s",
        task_id,
        mode,
        count,
        THRESHOLD,
    )

    if count < THRESHOLD:
        logger.info(
            "[TRAINING-TRIGGER] Task %s mode=%s below threshold, not retraining yet",
            task_id,
            mode,
        )
        return

    lock = training_lock(task_id, mode=mode)
    if not lock.acquire(blocking=False):
        logger.warning(
            "[TRAINING-TRIGGER] Task %s mode=%s retraining already in progress, skipping",
            task_id,
            mode,
        )
        return

    queue = django_rq.get_queue(settings.CVAT_QUEUES.IMPORT_DATA.value)
    if mode == "classification":
        queue.enqueue(retrain_classification_task_model, task_id)
    else:
        queue.enqueue(retrain_detection_task_model, task_id)

    logger.info(
        "[TRAINING-TRIGGER] Task %s mode=%s threshold reached, enqueued retraining",
        task_id,
        mode,
    )


def on_annotation_saved(*, task_id: int, shape_frames: int, tag_frames: int):
    """
    Called whenever new annotated frames are added to a task.

    shape_frames : newly supervised frames via shapes (object detection)
    tag_frames   : newly supervised frames via tags (classification)
    """

    _trigger_mode(task_id=task_id, frames=shape_frames, mode="detection")
    _trigger_mode(task_id=task_id, frames=tag_frames, mode="classification")
