import logging
import shutil
from pathlib import Path

import requests
from django.conf import settings

from cvat.apps.engine.models import Task

from .dataset import (
    clear_yolo_cache,
    ensure_data_yaml,
    remove_unlabeled_images,
    unpack_yolo_classification_dataset,
    unpack_yolo_dataset,
)
from .exporter import export_task_to_imagenet_classification, export_task_to_yolo
from .model_paths import ensure_task_dirs, yolo8cls_task_data_dir, yolo_task_data_dir

log = logging.getLogger(__name__)


def _resolve_classification_dataset_dir(data_dir: Path) -> Path:
    # Case 1: expected Ultralytics layout already present.
    if (data_dir / "train").is_dir():
        return data_dir

    # Case 2: a single nested folder contains train/.
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and (child / "train").is_dir():
            return child

    # Case 3: ImageNet-style export (class folders at root or in one nested folder).
    def _class_dirs(root: Path) -> list[Path]:
        ignore = {"train", "val", "test", "__MACOSX"}
        return [p for p in root.iterdir() if p.is_dir() and p.name not in ignore]

    candidate_roots = [data_dir] + [p for p in data_dir.iterdir() if p.is_dir()]
    for root in candidate_roots:
        classes = _class_dirs(root)
        if classes:
            train_dir = root / "train"
            train_dir.mkdir(exist_ok=True)
            for cls_dir in classes:
                target = train_dir / cls_dir.name
                if target.exists():
                    shutil.rmtree(target)
                cls_dir.rename(target)
            return root

    raise RuntimeError(f"Failed to locate classification dataset root under {data_dir}")


def retrain_detection_task_model(task_id: int):
    log.info("[TRAINER] Starting detection retraining for task %s", task_id)

    ensure_task_dirs(task_id)
    data_dir = yolo_task_data_dir(task_id)

    zip_path = export_task_to_yolo(task_id)
    unpack_yolo_dataset(zip_path, data_dir)
    remove_unlabeled_images(data_dir)
    clear_yolo_cache(data_dir)
    log.info("[TRAINER] Cleared YOLO detection cache for task %s", task_id)

    task = Task.objects.get(id=task_id)
    labels = task.project.label_set.all() if task.project_id else task.label_set.all()
    class_names = [l.name for l in labels]
    ensure_data_yaml(data_dir, class_names)

    # Keep YOLOv7 service model indexing behavior based on task.data.id.
    service_task_id = task.data.id
    resp = requests.post(
        f"{settings.YOLOV7_SERVICE['URL']}/train",
        json={"task_id": service_task_id},
        timeout=10,
    )
    resp.raise_for_status()

    log.info(
        "[TRAINER] YOLOv7 detection training triggered for task %s (service_task_id=%s)",
        task_id,
        service_task_id,
    )


def retrain_classification_task_model(task_id: int):
    log.info("[TRAINER] Starting classification retraining for task %s", task_id)

    ensure_task_dirs(task_id)
    data_dir = yolo8cls_task_data_dir(task_id)

    zip_path = export_task_to_imagenet_classification(task_id)
    unpack_yolo_classification_dataset(zip_path, data_dir)
    dataset_dir = _resolve_classification_dataset_dir(data_dir)
    log.info("[TRAINER] Classification dataset root for task %s: %s", task_id, dataset_dir)

    resp = requests.post(
        f"{settings.YOLOV8CLS_SERVICE['URL']}/train",
        json={"task_id": task_id, "dataset_dir": str(dataset_dir), "epochs": 30, "imgsz": 224},
        timeout=10,
    )
    resp.raise_for_status()

    log.info("[TRAINER] YOLOv8-CLS training triggered for task %s", task_id)


def retrain_task_model(task_id: int):
    # Backward compatibility: default old trigger path to classification.
    retrain_classification_task_model(task_id)
