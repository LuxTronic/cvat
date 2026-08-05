import logging
from pathlib import Path

from cvat.apps.dataset_manager.task import export_task

from .model_paths import yolo8cls_task_data_dir, yolo_task_data_dir

log = logging.getLogger(__name__)


def export_task_to_yolo(task_id: int) -> Path:
    """
    Export task annotations + images to YOLO ZIP.
    """
    data_dir = yolo_task_data_dir(task_id)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "dataset.zip"

    log.info("[EXPORT] Exporting task %s to %s", task_id, zip_path)

    export_task(
        task_id=task_id,
        dst_file=str(zip_path),
        format_name="Ultralytics YOLO Detection 1.0",
        save_images=True,
    )

    if not zip_path.exists():
        raise RuntimeError(f"Export failed, zip not found at {zip_path}")

    return zip_path


def export_task_to_yolo_classification(task_id: int) -> Path:
    """
    Export task annotations + images to ImageNet ZIP.
    """
    data_dir = yolo8cls_task_data_dir(task_id)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "dataset_classification.zip"

    log.info("[EXPORT] Exporting task %s to %s (classification)", task_id, zip_path)

    export_task(
        task_id=task_id,
        dst_file=str(zip_path),
        format_name="ImageNet 1.0",
        save_images=True,
    )

    if not zip_path.exists():
        raise RuntimeError(f"Classification export failed, zip not found at {zip_path}")

    return zip_path


def export_task_to_imagenet_classification(task_id: int) -> Path:
    # Explicit alias used by the classification training worker.
    return export_task_to_yolo_classification(task_id)
