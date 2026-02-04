from cvat.apps.engine.training.counters import increment_task_annotation_counter


def handle_new_annotations(*, job_id: int, frames: int):
    """
    Called AFTER annotations are saved.
    `frames` = number of frames that transitioned from 0 → >=1 shapes.
    """

    if frames <= 0:
        return

    # Increment task-level counter
    increment_task_annotation_counter(
        task_id=job_id,
        frames=frames,
    )
