import subprocess
from pathlib import Path
from django.conf import settings
from cvat.apps.engine.training.counters import reset_counter

YOLO_ROOT = Path("/yolov7")
MODELS_ROOT = Path("/models")

def retrain_task_model(task_id: int):
    task_root = MODELS_ROOT / f"task_{task_id}"
    data_dir = task_root / "data"

    task_root.mkdir(parents=True, exist_ok=True)

    versions = sorted(p for p in task_root.glob("v*") if p.is_dir())
    next_version = f"v{len(versions) + 1}"
    out_dir = task_root / next_version

    active = task_root / "active"
    if active.exists():
        weights = active / "weights" / "best.pt"
    else:
        weights = YOLO_ROOT / "yolov7.pt"

    cmd = [
        "python", "train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", "10",
        "--data", str(data_dir / "data.yaml"),
        "--weights", str(weights),
        "--project", str(task_root),
        "--name", next_version,
    ]

    subprocess.check_call(cmd, cwd=YOLO_ROOT)

    # promote atomically
    active.unlink(missing_ok=True)
    active.symlink_to(out_dir)

    reset_counter(task_id)
