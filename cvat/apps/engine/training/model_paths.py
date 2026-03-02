from pathlib import Path

MODELS_ROOT = Path("/models")

def task_root(task_id: int) -> Path:
    return MODELS_ROOT / f"task_{task_id}"

def task_data_dir(task_id: int) -> Path:
    return task_root(task_id) / "data"

def ensure_task_dirs(task_id: int):
    """
    Ensure only the base directories.
    Dataset structure is created by CVAT exporter.
    """
    task_data_dir(task_id).mkdir(parents=True, exist_ok=True)
