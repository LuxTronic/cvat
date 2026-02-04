import django_rq
from cvat.apps.engine.training.counters import (
    increment_task_annotation_counter,
    should_retrain,
    training_lock,
)
from cvat.apps.engine.training.retrain_worker import retrain_task_model
from django.conf import settings

def on_annotation_saved(task_id: int, frames: int = 1):
    count = increment_task_annotation_counter(task_id, frames)

    if count < 20:
        return

    lock = training_lock(task_id)
    if not lock.acquire(blocking=False):
        return

    queue = django_rq.get_queue(settings.CVAT_QUEUES.IMPORT_DATA.value)
    queue.enqueue(retrain_task_model, task_id)
