from .model_paths import ensure_task_dirs, task_data_dir
from .exporter import export_task_to_yolo
from .dataset import clear_yolo_cache, split_train_val, unpack_yolo_dataset, clean_orphan_labels
from .dataset import ensure_data_yaml
from cvat.apps.engine.models import Task
from cvat.apps.engine.training.counters import reset
import logging
import requests
from django.conf import settings

log = logging.getLogger(__name__)

def retrain_task_model(task_id: int):
    log.info("[TRAINER] Starting retraining for task %s", task_id)

    ensure_task_dirs(task_id)
    data_dir = task_data_dir(task_id)

    # 1. Export ZIP
    zip_path = export_task_to_yolo(task_id)

    # 2. Unpack ZIP
    unpack_yolo_dataset(zip_path, data_dir)

    split_train_val(data_dir, val_ratio=0.2)
    log.info("[TRAINER] Splitting train/val for task %s", task_id)

    clean_orphan_labels(data_dir)
    log.info("[TRAINER] Cleaned orphan labels for task %s", task_id)

    clear_yolo_cache(data_dir)
    log.info("[TRAINER] Cleared YOLO cache for task %s", task_id)

    # 3. Generate data.yaml
    task = Task.objects.get(id=task_id)
    labels = (
        task.project.label_set.all()
        if task.project_id
        else task.label_set.all()
    )
    class_names = [l.name for l in labels]
    ensure_data_yaml(data_dir, class_names)

    # 4. Trigger YOLO service
    resp = requests.post(
        f"{settings.YOLOV7_SERVICE['URL']}/train",
        json={"task_id": task_id},
        timeout=10,
    )
    resp.raise_for_status()

    log.info("[TRAINER] YOLO training triggered for task %s", task_id)
