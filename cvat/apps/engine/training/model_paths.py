from pathlib import Path

MODELS_ROOT = Path("/models")
YOLOV7_ROOT = MODELS_ROOT / "yolov7"
YOLOV8CLS_ROOT = MODELS_ROOT / "yolov8_cls"

def yolo_task_root(task_id: int) -> Path:
    return YOLOV7_ROOT / f"task_{task_id}"

def yolo_task_data_dir(task_id: int) -> Path:
    return yolo_task_root(task_id) / "data"

def yolo8cls_task_root(task_id: int) -> Path:
    return YOLOV8CLS_ROOT / f"task_{task_id}"

def yolo8cls_task_data_dir(task_id: int) -> Path:
    return yolo8cls_task_root(task_id) / "data"

# Backward compatibility for existing imports.
def task_root(task_id: int) -> Path:
    return yolo8cls_task_root(task_id)

def task_data_dir(task_id: int) -> Path:
    return yolo8cls_task_data_dir(task_id)

def ensure_task_dirs(task_id: int):
    """
    Ensure only the base directories.
    Dataset structure is created by CVAT exporter.
    """
    yolo_task_data_dir(task_id).mkdir(parents=True, exist_ok=True)
    yolo8cls_task_data_dir(task_id).mkdir(parents=True, exist_ok=True)
