import logging
import random
import shutil
from pathlib import Path

import requests
from django.conf import settings

from cvat.apps.engine.models import Task

from .counters import reset
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
    def _class_dirs(root: Path) -> list[Path]:
        ignore = {"train", "val", "test", "__MACOSX", "no_label"}
        return [p for p in root.iterdir() if p.is_dir() and p.name not in ignore]

    def _is_image_file(path: Path) -> bool:
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _prepare_imagenet_split(root: Path, classes: list[Path]) -> Path:
        # Remove explicit no_label buckets if present.
        for name in ("no_label", "NO_LABEL"):
            no_label_dir = root / name
            if no_label_dir.exists() and no_label_dir.is_dir():
                shutil.rmtree(no_label_dir)

        train_dir = root / "train"
        val_dir = root / "val"
        if train_dir.exists():
            shutil.rmtree(train_dir)
        if val_dir.exists():
            shutil.rmtree(val_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(42)

        for cls_dir in sorted(classes, key=lambda p: p.name):
            image_paths = [p for p in cls_dir.rglob("*") if p.is_file() and _is_image_file(p)]
            if not image_paths:
                continue

            rng.shuffle(image_paths)
            total = len(image_paths)
            val_count = int(total * 0.15)
            if total > 1:
                val_count = max(1, val_count)
                val_count = min(val_count, total - 1)
            else:
                val_count = 0

            for idx, src in enumerate(image_paths):
                split_root = val_dir if idx < val_count else train_dir
                dst = split_root / cls_dir.name / src.relative_to(cls_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

            shutil.rmtree(cls_dir, ignore_errors=True)

        return root

    def _ensure_val_from_train(root: Path) -> Path:
        train_dir = root / "train"
        val_dir = root / "val"

        # Remove no_label if present in train.
        for name in ("no_label", "NO_LABEL"):
            no_label_dir = train_dir / name
            if no_label_dir.exists() and no_label_dir.is_dir():
                shutil.rmtree(no_label_dir)

        val_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(42)

        class_dirs = [p for p in train_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
        for cls_dir in sorted(class_dirs, key=lambda p: p.name):
            image_paths = [p for p in cls_dir.rglob("*") if p.is_file() and _is_image_file(p)]
            if not image_paths:
                continue

            rng.shuffle(image_paths)
            total = len(image_paths)
            val_count = int(total * 0.15)
            if total > 1:
                val_count = max(1, val_count)
                val_count = min(val_count, total - 1)
            else:
                val_count = 0

            for idx, src in enumerate(image_paths):
                if idx >= val_count:
                    break
                dst = val_dir / cls_dir.name / src.relative_to(cls_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

        return root

    candidate_roots = [data_dir] + [p for p in data_dir.iterdir() if p.is_dir()]
    for root in candidate_roots:
        # If this root already has train/, ensure val/ split exists.
        if (root / "train").is_dir():
            return _ensure_val_from_train(root)

        classes = _class_dirs(root)
        if classes:
            return _prepare_imagenet_split(root, classes)

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
    reset(task_id, mode="detection")

    log.info(
        "[TRAINER] YOLOv7 detection training triggered for task %s (service_task_id=%s), counter reset",
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
    reset(task_id, mode="classification")

    log.info("[TRAINER] YOLOv8-CLS training triggered for task %s, counter reset", task_id)


def retrain_task_model(task_id: int):
    # Backward compatibility: default old trigger path to classification.
    retrain_classification_task_model(task_id)
