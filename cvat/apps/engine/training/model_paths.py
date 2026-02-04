from pathlib import Path

MODELS_ROOT = Path("/models")

def task_root(task_id: int) -> Path:
    return MODELS_ROOT / f"task_{task_id}"

def task_data_dir(task_id: int) -> Path:
    return task_root(task_id) / "data"

def task_versions_dir(task_id: int) -> Path:
    return task_root(task_id)

def task_active_model(task_id: int) -> Path:
    return task_root(task_id) / "active"

def ensure_task_dirs(task_id: int):
    root = task_root(task_id)
    root.mkdir(parents=True, exist_ok=True)
    task_data_dir(task_id).mkdir(exist_ok=True)
