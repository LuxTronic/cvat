import logging

from django.db import transaction

from cvat.apps.engine.models import Job

logger = logging.getLogger(__name__)


def handle_new_annotations(*, job_id: int, shape_frames: int, tag_frames: int):
    """
    Schedule the retraining trigger for after the annotation write commits.

    Callers run inside ``transaction.atomic()``, so this must not act on the
    annotations directly: a rollback after the fact would leave a retraining job
    enqueued for annotations that never existed. ``transaction.on_commit`` defers
    the work until the outermost transaction actually commits, and runs
    immediately when there is no transaction in progress.

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

    transaction.on_commit(
        lambda: _trigger_retraining(job_id=job_id, shape_frames=shape_frames, tag_frames=tag_frames)
    )


def _trigger_retraining(*, job_id: int, shape_frames: int, tag_frames: int) -> None:
    # Retraining is an optimisation, never a reason to fail a save the user has
    # already been told succeeded, so nothing here is allowed to propagate.
    try:
        db_job = Job.objects.select_related("segment__task").get(id=job_id)
        task_id = db_job.segment.task.id

        logger.info("[TRAINING-HOOK] Resolved job %s -> task %s", job_id, task_id)

        from cvat.apps.engine.training.trigger import on_annotation_saved

        on_annotation_saved(task_id=task_id, shape_frames=shape_frames, tag_frames=tag_frames)
    except Exception:
        logger.exception(
            "[TRAINING-HOOK] Retraining trigger failed for job %s; annotations are unaffected",
            job_id,
        )
