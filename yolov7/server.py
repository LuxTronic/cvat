import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import subprocess
import torch
import json
import logging

app = FastAPI()

import os
import redis
import logging


REDIS_HOST = os.environ.get("CVAT_REDIS_INMEM_HOST", "cvat_redis_inmem")
REDIS_PORT = int(os.environ.get("CVAT_REDIS_INMEM_PORT", "6379"))

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)

def _run_training(cmd, task_id: int):
    try:
        log.info("[YOLO] Training started for task %s", task_id)

        subprocess.check_call(cmd, cwd=YOLO_ROOT)

        log.info("[YOLO] Training completed for task %s", task_id)

        # ✅ SUCCESS → reset counter + release lock
        r.delete(f"task:{task_id}:frames")
        r.delete(f"task:{task_id}:training")

        log.info(
            "[YOLO] Reset counters & training lock for task %s",
            task_id,
        )

    except Exception:
        log.exception("[YOLO] Training FAILED for task %s", task_id)

        r.delete(f"task:{task_id}:training")

        raise

DEVICE = "0" if torch.cuda.is_available() else  "cpu"
YOLO_ROOT = Path("/yolov7")
MODELS_ROOT = Path("/models")
log = logging.getLogger("yolo-trainer")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    task_id: int = 0,
):
    """
    Run YOLOv7 inference on a single image.
    Returns detections in YOLO normalized format.
    """ 
    weights_path = Path(f"/models/task_{task_id}/active/weights/best.pt")

    if not weights_path.exists():
        return JSONResponse(
            status_code=400,
            content={"error": f"No active model for task {task_id}"}
    )


    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save uploaded image
        image_path = tmpdir / image.filename
        image_path.write_bytes(await image.read())

        # Output directory
        out_dir = tmpdir / "runs"

        # Run YOLOv7 detect.py
        cmd = [
            "python", "detect.py",
            "--weights", str(weights_path),
            "--source", str(image_path),
            "--device", DEVICE,
            "--workers", "0",        
            "--save-txt",
            "--save-conf",
            "--conf", "0.01",
            "--project", str(out_dir),
            "--name", "pred",
            "--exist-ok",
            "--nosave"
        ]

        subprocess.run(cmd, check=True)

        # Parse YOLO output
        labels_dir = out_dir / "pred" / "labels"
        detections = []

        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                for line in label_file.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else None

                    detections.append({
                        "class_id": class_id,
                        "xc": xc,
                        "yc": yc,
                        "w": w,
                        "h": h,
                        "confidence": conf,
                    })

        return JSONResponse(content=detections)



@app.post("/train")
def train_task(payload: dict):
    task_id = payload["task_id"]

    task_dir = MODELS_ROOT / f"task_{task_id}"
    data_yaml = task_dir / "data" / "data.yaml"

    cmd = [
        "python", "train.py",
        "--img", "640",
        "--batch", "8",
        "--epochs", "30",
        "--data", str(data_yaml),
        "--weights", "/yolov7/weights/yolov7.pt",
        "--project", str(task_dir),
        "--name", f"v{len(list(task_dir.glob('v*'))) + 1}",
    ]

    threading.Thread(
        target=_run_training,
        args=(cmd, task_id),
        daemon=True,
    ).start()

    return {
        "status": "started",
        "task_id": task_id,
    }
