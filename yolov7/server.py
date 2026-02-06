import os
import threading
import subprocess
import logging
from pathlib import Path
from typing import Dict

import redis
import torch
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# ----------------------------
# App / Logging
# ----------------------------
app = FastAPI(debug=True)
log = logging.getLogger("yolov7")
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Paths / Redis
# ----------------------------
YOLO_ROOT = Path("/yolov7")
MODELS_ROOT = Path("/models")

REDIS_HOST = os.environ.get("CVAT_REDIS_INMEM_HOST", "cvat_redis_inmem")
REDIS_PORT = int(os.environ.get("CVAT_REDIS_INMEM_PORT", "6379"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ----------------------------
# Device
# ----------------------------
DEVICE = select_device("0" if torch.cuda.is_available() else "cpu")
USE_HALF = (DEVICE.type != "cpu")

# ----------------------------
# Model cache (task_id -> model)
# ----------------------------
MODEL_CACHE: Dict[int, torch.nn.Module] = {}
MODEL_CACHE_LOCK = threading.Lock()

# Optional: prevent two threads running inference simultaneously on same task model
MODEL_INFER_LOCKS: Dict[int, threading.Lock] = {}
MODEL_INFER_LOCKS_LOCK = threading.Lock()


def _get_infer_lock(task_id: int) -> threading.Lock:
    with MODEL_INFER_LOCKS_LOCK:
        if task_id not in MODEL_INFER_LOCKS:
            MODEL_INFER_LOCKS[task_id] = threading.Lock()
        return MODEL_INFER_LOCKS[task_id]


def get_model_for_task(task_id: int) -> torch.nn.Module:
    # Load once per task_id, keep in memory
    with MODEL_CACHE_LOCK:
        if task_id in MODEL_CACHE:
            return MODEL_CACHE[task_id]

        weights_path = MODELS_ROOT / f"task_{task_id}" / "active" / "weights" / "best.pt"
        if not weights_path.exists():
            raise RuntimeError(f"No active model for task {task_id}: {weights_path} does not exist")

        log.info("[YOLO] Loading model for task %s from %s", task_id, weights_path)

        model = attempt_load(str(weights_path), map_location=DEVICE)
        model.eval()

        if USE_HALF:
            model.half()

        MODEL_CACHE[task_id] = model
        return model


def invalidate_task_model(task_id: int) -> None:
    # Called after training switches "active" symlink
    with MODEL_CACHE_LOCK:
        if task_id in MODEL_CACHE:
            log.info("[YOLO] Invalidating cached model for task %s", task_id)
            MODEL_CACHE.pop(task_id, None)


def _run_training(cmd, task_id: int):
    try:
        log.info("[YOLO] Training started for task %s", task_id)
        subprocess.check_call(cmd, cwd=str(YOLO_ROOT))
        log.info("[YOLO] Training completed for task %s", task_id)

        task_dir = MODELS_ROOT / f"task_{task_id}"

        versions = sorted(
            [p for p in task_dir.glob("v*") if p.is_dir()],
            key=lambda p: p.name,
        )
        if not versions:
            raise RuntimeError(f"No trained versions found for task {task_id}")

        latest = versions[-1]
        active = task_dir / "active"

        if active.exists() or active.is_symlink():
            active.unlink()
        active.symlink_to(latest)

        log.info("[YOLO] Activated model for task %s → %s", task_id, latest)

        # IMPORTANT: clear cached model so next infer loads new weights
        invalidate_task_model(task_id)

        # reset redis counters/locks
        r.delete(f"task:{task_id}:frames")
        r.delete(f"task:{task_id}:training")
        log.info("[YOLO] Reset counters & training lock for task %s", task_id)

    except Exception:
        log.exception("[YOLO] Training FAILED for task %s", task_id)
        r.delete(f"task:{task_id}:training")
        raise


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "use_half": USE_HALF,
    }


@app.post("/infer")
async def infer(image: UploadFile = File(...), task_id: int = 0):
    try:
        model = get_model_for_task(task_id)
        infer_lock = _get_infer_lock(task_id)

        # Load image from request (no disk)
        img = Image.open(image.file).convert("RGB")
        img0 = np.array(img)  # original HWC RGB
        h0, w0 = img0.shape[:2]

        # YOLOv7 expects 640x640 by default (unless you trained different)
        img_resized = img.resize((640, 640))
        img_np = np.array(img_resized)

        img_tensor = torch.from_numpy(img_np).to(DEVICE)
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()  # HWC -> CHW
        img_tensor = img_tensor.float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # add batch

        if USE_HALF:
            img_tensor = img_tensor.half()

        with infer_lock:
            with torch.no_grad():
                pred = model(img_tensor)[0]
                pred = non_max_suppression(pred, conf_thres=0.02, iou_thres=0.10)

        detections = []

        for det in pred:
            if det is None or len(det) == 0:
                continue

            # scale boxes back to original image size
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(float, xyxy)

                detections.append({
                    "class_id": int(cls),
                    "confidence": float(conf),
                    "xc": ((x1 + x2) / 2.0) / w0,
                    "yc": ((y1 + y2) / 2.0) / h0,
                    "w": (x2 - x1) / w0,
                    "h": (y2 - y1) / h0,
                })

        return JSONResponse(content=detections)

    except Exception as e:
        log.exception("[YOLO-INFER] Failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


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

    threading.Thread(target=_run_training, args=(cmd, task_id), daemon=True).start()

    return {"status": "started", "task_id": task_id}
