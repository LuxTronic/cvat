import logging

from cvat.apps.engine.models import Job

logger = logging.getLogger(__name__)


def handle_new_annotations(*, job_id: int, shape_frames: int, tag_frames: int):
    """
    Called AFTER annotations are committed to the database.

    job_id       : Job where annotations were saved
    shape_frames : Number of frames that gained new shape annotations
    tag_frames   : Number of frames that gained new tag annotations
    """
    logger.info(
        "[TRAINING-HOOK] handle_new_annotations(job_id=%s, shape_frames=%s, tag_frames=%s)",
        job_id,
        shape_frames,
        tag_frames,
    )

    if shape_frames <= 0 and tag_frames <= 0:
        logger.info("[TRAINING-HOOK] No new frames, skipping")
        return

    db_job = Job.objects.select_related("segment__task").get(id=job_id)
    task_id = db_job.segment.task.id

    logger.info("[TRAINING-HOOK] Resolved job %s -> task %s", job_id, task_id)

    from cvat.apps.engine.training.trigger import on_annotation_saved

    on_annotation_saved(task_id=task_id, shape_frames=shape_frames, tag_frames=tag_frames)
