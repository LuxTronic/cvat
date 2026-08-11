import os
import re
import threading
import subprocess
import logging
import time
import uuid
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import boto3
import redis
import torch
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.datasets import letterbox

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
YOLO_MODELS_ROOT = MODELS_ROOT / "yolov7"

REDIS_HOST = os.environ.get("CVAT_REDIS_INMEM_HOST", "cvat_redis_inmem")
REDIS_PORT = int(os.environ.get("CVAT_REDIS_INMEM_PORT", "6379"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ----------------------------
# Device
# ----------------------------
DEVICE = select_device("0" if torch.cuda.is_available() else "cpu")
USE_HALF = (DEVICE.type != "cpu")

# ----------------------------
# Model cache (task-local active model or selected legacy model -> model)
# ----------------------------
MODEL_CACHE: Dict[str, torch.nn.Module] = {}
MODEL_CACHE_LOCK = threading.Lock()

# Optional: prevent two threads running inference simultaneously on same task model
MODEL_INFER_LOCKS: Dict[str, threading.Lock] = {}
MODEL_INFER_LOCKS_LOCK = threading.Lock()


def _get_infer_lock(cache_key: str) -> threading.Lock:
    with MODEL_INFER_LOCKS_LOCK:
        if cache_key not in MODEL_INFER_LOCKS:
            MODEL_INFER_LOCKS[cache_key] = threading.Lock()
        return MODEL_INFER_LOCKS[cache_key]


def _safe_model_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "selected_model").strip("_") or "selected_model"


# Weights are unpickled by attempt_load(), so loading a checkpoint executes
# whatever the file contains. Restrict where they may come from: an unrestricted
# model_uri would let any caller on this network run code in this container.
ALLOWED_MODEL_BUCKETS = {
    bucket.strip()
    for bucket in os.environ.get("MODEL_BUCKET_ALLOWLIST", "luxmodels").split(",")
    if bucket.strip()
}
ALLOWED_MODEL_KEY_PREFIXES = tuple(
    prefix.strip()
    for prefix in os.environ.get("MODEL_KEY_PREFIX_ALLOWLIST", "Deployed_Models/").split(",")
    if prefix.strip()
)


def _parse_s3_uri(model_uri: str):
    parsed = urlparse(model_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise RuntimeError(f"Unsupported model URI: {model_uri}")

    bucket, key = parsed.netloc, parsed.path.lstrip("/")

    if bucket not in ALLOWED_MODEL_BUCKETS:
        raise RuntimeError(f"Model bucket is not allowlisted: {bucket}")
    if ALLOWED_MODEL_KEY_PREFIXES and not key.startswith(ALLOWED_MODEL_KEY_PREFIXES):
        raise RuntimeError(f"Model key is not under an allowlisted prefix: {key}")

    return bucket, key


def _download_s3_model(model_uri: str, weights_path: Path) -> None:
    bucket, key = _parse_s3_uri(model_uri)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = weights_path.with_suffix(weights_path.suffix + ".tmp")

    log.info("[YOLO] Downloading selected model | s3=%s | target=%s", model_uri, weights_path)
    boto3.client("s3").download_file(bucket, key, str(tmp_path))
    tmp_path.replace(weights_path)


def selected_model_weights_path(model_id: str, model_uri: str) -> Path:
    safe_id = _safe_model_id(model_id or model_uri)
    return YOLO_MODELS_ROOT / "selected" / safe_id / "weights" / "best.pt"


def get_model_from_weights(cache_key: str, weights_path: Path) -> torch.nn.Module:
    # Load once per selected key, keep in memory
    with MODEL_CACHE_LOCK:
        if cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key]

        if not weights_path.exists():
            raise RuntimeError(f"No YOLOv7 model weights found for {cache_key}: {weights_path} does not exist")

        version_dir = weights_path.parents[2].name
        log.info(
            "[YOLO] Loading model | key=%s | version=%s | weights=%s | device=%s | half=%s",
            cache_key,
            version_dir,
            weights_path,
            DEVICE,
            USE_HALF,
        )
        model = attempt_load(str(weights_path), map_location=DEVICE)
        model.eval()

        if USE_HALF:
            model.half()

        MODEL_CACHE[cache_key] = model
        return model


def get_model_for_task(task_id: int) -> torch.nn.Module:
    weights_path = YOLO_MODELS_ROOT / f"task_{task_id}" / "active" / "weights" / "best.pt"
    return get_model_from_weights(f"task:{task_id}", weights_path)


def get_selected_model(task_id: int, model_id: str, model_uri: str):
    if not model_uri:
        cache_key = f"task:{task_id}"
        weights_path = YOLO_MODELS_ROOT / f"task_{task_id}" / "active" / "weights" / "best.pt"
        return cache_key, weights_path, get_model_from_weights(cache_key, weights_path)

    cache_key = f"selected:{_safe_model_id(model_id or model_uri)}"
    weights_path = selected_model_weights_path(model_id, model_uri)
    if not weights_path.exists():
        _download_s3_model(model_uri, weights_path)
    return cache_key, weights_path, get_model_from_weights(cache_key, weights_path)


def invalidate_task_model(task_id: int) -> None:
    # Called after training switches "active" symlink
    with MODEL_CACHE_LOCK:
        cache_key = f"task:{task_id}"
        if cache_key in MODEL_CACHE:
            log.info("[YOLO] Invalidating cached model for task %s", task_id)
            MODEL_CACHE.pop(cache_key, None)


def _run_training(cmd, task_id: int):
    try:
        log.info("[YOLO] Training started for task %s", task_id)
        subprocess.check_call(cmd, cwd=str(YOLO_ROOT))
        log.info("[YOLO] Training completed for task %s", task_id)

        task_dir = YOLO_MODELS_ROOT / f"task_{task_id}"

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
async def infer(
    image: UploadFile = File(...),
    task_id: int = 0,
    model_id: str = "",
    model_uri: str = "",
):
    try:
        # ----------------------------
        # Load model (cached per task)
        # ----------------------------
        cache_key, weights_path, model = get_selected_model(task_id, model_id, model_uri)
        infer_lock = _get_infer_lock(cache_key)

        version = weights_path.parents[2].name
        stride = int(model.stride.max())

        # ----------------------------
        # Logging: model + request
        # ----------------------------
        log.info(
            "[YOLO-INFER] task=%s | model=%s | version=%s | weights=%s | stride=%d | half=%s",
            task_id,
            model_id or "task-active",
            version,
            weights_path,
            stride,
            USE_HALF,
        )

        log.info(
            "[YOLO-INFER] request | filename=%s | content_type=%s",
            image.filename,
            image.content_type,
        )

        # ----------------------------
        # Load image
        # ----------------------------
        img = Image.open(image.file).convert("RGB")
        img0 = np.array(img)  # original HWC RGB
        h0, w0 = img0.shape[:2]

        # ----------------------------
        # Inference image size (MATCH TRAINING)
        # ----------------------------
        TRAIN_IMG_SIZE = 512  # must match --img-size used in training
        IMG_SIZE = int(np.ceil(TRAIN_IMG_SIZE / stride) * stride)

        log.info(
            "[YOLO-INFER] image | original=%dx%d | resized=%dx%d",
            w0,
            h0,
            IMG_SIZE,
            IMG_SIZE,
        )

        # ----------------------------
        # Resize (simple resize; letterbox optional)
        # ----------------------------
        # ----------------------------
        # Letterbox resize (YOLO-correct)
        # ----------------------------
        img_lb, ratio, pad = letterbox(
            img0,
            new_shape=IMG_SIZE,
            auto=False,
            scaleFill=False,
        )

        img_tensor = torch.from_numpy(img_lb).to(DEVICE)
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()
        img_tensor = img_tensor.float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)


        if USE_HALF:
            img_tensor = img_tensor.half()

        # ----------------------------
        # Inference
        # ----------------------------
        with infer_lock:
            with torch.no_grad():
                pred = model(img_tensor)[0]
                pred = non_max_suppression(
                    pred,
                    conf_thres=0.20,
                    iou_thres=0.10,
                )

        # ----------------------------
        # Post-processing
        # ----------------------------
        detections = []

        for det in pred:
            if det is None or len(det) == 0:
                continue

            # scale boxes back to original image size
            det[:, :4] = scale_coords(
                img_lb.shape[:2], det[:, :4], img0.shape
            ).round()
            log.info(
                "[YOLO-INFER] letterbox | ratio=%.4f | pad=(%d,%d)",
                ratio[0],
                pad[0],
                pad[1],
            )


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

        log.info(
            "[YOLO-INFER] result | task=%s | detections=%d",
            task_id,
            len(detections),
        )

        return JSONResponse(content=detections)

    except Exception as e:
        log.exception("[YOLO-INFER] Failed")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/train")
def train_task(payload: dict):
    # Coerce before the value reaches a path: "../../../etc" would otherwise
    # escape the model tree and land training artifacts anywhere writable.
    try:
        task_id = int(payload["task_id"])
    except (KeyError, TypeError, ValueError):
        return JSONResponse(
            status_code=400, content={"error": "task_id is required and must be an integer"}
        )

    task_dir = YOLO_MODELS_ROOT / f"task_{task_id}"
    data_yaml = task_dir / "data" / "data.yaml"

    if not data_yaml.exists():
        return JSONResponse(
            status_code=400, content={"error": f"No dataset for task {task_id}: {data_yaml} missing"}
        )

    cmd = [
        "python", "train.py",
        "--weights", "/yolov7/weights/yolov7.pt",
        "--img-size", "512",
        "--batch-size", "8",
        "--epochs", "100",
        "--data", str(data_yaml),
        "--freeze", "10",
        "--rect",
        "--hyp", "/yolov7/data/hyp.scratch.tiny.yaml",
        "--workers", "8",
        "--notest",
        "--project", str(task_dir),
        # A directory count races when two /train calls arrive together; the
        # submit timestamp keeps concurrent runs in separate directories.
        "--name", f"v{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
    ]



    threading.Thread(target=_run_training, args=(cmd, task_id), daemon=True).start()

    return {"status": "started", "task_id": task_id}
