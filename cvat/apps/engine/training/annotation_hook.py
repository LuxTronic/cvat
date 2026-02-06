import logging
from cvat.apps.engine.models import Job

logger = logging.getLogger(__name__)


def handle_new_annotations(*, job_id: int, frames: int):
    """
    Called AFTER annotations are committed to the database.

    job_id  : Job where annotations were saved
    frames  : Number of frames that transitioned from 0 → >=1 annotations
    """
    logger.info(
        "[TRAINING-HOOK] handle_new_annotations(job_id=%s, frames=%s)",
        job_id,
        frames,
    )

    if frames <= 0:
        logger.info("[TRAINING-HOOK] No new frames, skipping")
        return

    # Resolve job → task
    db_job = Job.objects.select_related("segment__task").get(id=job_id)
    task_id = db_job.segment.task.data.id

    logger.info(
        "[TRAINING-HOOK] Resolved job %s → task %s",
        job_id,
        task_id,
    )

    # Delegate to training policy
    from cvat.apps.engine.training.trigger import on_annotation_saved
    on_annotation_saved(task_id=task_id, frames=frames)
