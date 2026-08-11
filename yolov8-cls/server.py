from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import redis
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(debug=True)
log = logging.getLogger("yolov8-cls")
logging.basicConfig(level=logging.INFO)

YOLO_ROOT = Path("/yolov8-cls")
MODELS_ROOT = Path("/models")
YOLO8_MODELS_ROOT = MODELS_ROOT / "yolov8_cls"
BASE_WEIGHTS = YOLO_ROOT / "weights" / "yolov8n-cls.pt"

REDIS_HOST = os.environ.get("CVAT_REDIS_INMEM_HOST", "cvat_redis_inmem")
REDIS_PORT = int(os.environ.get("CVAT_REDIS_INMEM_PORT", "6379"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

MODEL_CACHE: Dict[int, YOLO] = {}
MODEL_CACHE_LOCK = threading.Lock()
MODEL_INFER_LOCKS: Dict[int, threading.Lock] = {}
MODEL_INFER_LOCKS_LOCK = threading.Lock()


def _get_infer_lock(task_id: int) -> threading.Lock:
    with MODEL_INFER_LOCKS_LOCK:
        if task_id not in MODEL_INFER_LOCKS:
            MODEL_INFER_LOCKS[task_id] = threading.Lock()
        return MODEL_INFER_LOCKS[task_id]


def _task_dir(task_id: int) -> Path:
    return YOLO8_MODELS_ROOT / f"task_{task_id}"


def _active_weights_path(task_id: int) -> Path:
    return _task_dir(task_id) / "active" / "weights" / "best.pt"


def _list_versions(task_id: int) -> List[Path]:
    tdir = _task_dir(task_id)
    return sorted([p for p in tdir.glob("v*") if p.is_dir()], key=lambda p: p.name)


def invalidate_task_model(task_id: int) -> None:
    with MODEL_CACHE_LOCK:
        if task_id in MODEL_CACHE:
            log.info("[YOLOv8-CLS] Invalidating cached model for task %s", task_id)
            MODEL_CACHE.pop(task_id, None)


def get_model_for_task(task_id: int) -> YOLO:
    with MODEL_CACHE_LOCK:
        if task_id in MODEL_CACHE:
            return MODEL_CACHE[task_id]

        weights_path = _active_weights_path(task_id)
        if not weights_path.exists():
            raise RuntimeError(f"No active model for task {task_id}: {weights_path}")

        version_dir = weights_path.parents[2].name
        log.info(
            "[YOLOv8-CLS] Loading model | task=%s | version=%s | weights=%s",
            task_id,
            version_dir,
            weights_path,
        )

        model = YOLO(str(weights_path))
        MODEL_CACHE[task_id] = model
        return model


def _activate_latest_model(task_id: int) -> Path:
    versions = _list_versions(task_id)
    if not versions:
        raise RuntimeError(f"No trained versions found for task {task_id}")

    latest = versions[-1]
    active = _task_dir(task_id) / "active"

    if active.exists() or active.is_symlink():
        active.unlink()
    active.symlink_to(latest)

    log.info("[YOLOv8-CLS] Activated model for task %s -> %s", task_id, latest)
    invalidate_task_model(task_id)
    return latest


def _run_training(task_id: int, dataset_dir: str, epochs: int, imgsz: int):
    try:
        task_dir = _task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        next_version = f"v{len(_list_versions(task_id)) + 1}"

        # Run training in a separate Python process for better isolation/logging.
        cmd = [
            "python",
            "-c",
            (
                "from ultralytics import YOLO; "
                f"model = YOLO(r'{str(BASE_WEIGHTS)}'); "
                f"model.train(data=r'{dataset_dir}', epochs={int(epochs)}, imgsz={int(imgsz)}, "
                f"project=r'{str(task_dir)}', name=r'{next_version}')"
            ),
        ]

        log.info(
            "[YOLOv8-CLS] Training started | task=%s | dataset=%s | epochs=%s | imgsz=%s | version=%s",
            task_id,
            dataset_dir,
            epochs,
            imgsz,
            next_version,
        )

        # Fixed argv assembled above, no shell.
        subprocess.check_call(cmd, cwd=str(YOLO_ROOT))  # nosec B603
        _activate_latest_model(task_id)

        r.delete(f"task:{task_id}:frames")
        r.delete(f"task:{task_id}:training")

        log.info("[YOLOv8-CLS] Training completed | task=%s", task_id)
    except Exception:
        log.exception("[YOLOv8-CLS] Training FAILED | task=%s", task_id)
        r.delete(f"task:{task_id}:training")
        raise


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "base_weights": str(BASE_WEIGHTS),
        "base_weights_exists": BASE_WEIGHTS.exists(),
    }


@app.post("/train")
def train(payload: Dict[str, Any]):
    task_id = int(payload["task_id"])
    dataset_dir = str(payload.get("dataset_dir") or (_task_dir(task_id) / "data"))
    epochs = int(payload.get("epochs", 30))
    imgsz = int(payload.get("imgsz", 224))

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return JSONResponse(
            status_code=400,
            content={"error": f"dataset_dir does not exist: {dataset_dir}"},
        )

    threading.Thread(
        target=_run_training,
        args=(task_id, dataset_dir, epochs, imgsz),
        daemon=True,
    ).start()

    return {
        "status": "started",
        "task_id": task_id,
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "imgsz": imgsz,
    }


@app.post("/infer")
async def infer(
    task_id: int,
    image: UploadFile = File(...),
    topk: int = 1,
):
    try:
        model = get_model_for_task(task_id)
        infer_lock = _get_infer_lock(task_id)

        pil_img = Image.open(image.file).convert("RGB")
        img_np = np.array(pil_img)

        with infer_lock:
            results = model.predict(source=img_np, verbose=False)

        probs = results[0].probs
        if probs is None:
            return JSONResponse(content=[])

        k = max(1, int(topk))
        top1_conf = float(probs.top1conf.item())
        top1_idx = int(probs.top1)

        out = [
            {
                "class_id": top1_idx,
                "class_name": results[0].names.get(top1_idx, str(top1_idx)),
                "confidence": top1_conf,
            }
        ]

        if k > 1:
            idxs = probs.top5[: min(k, 5)]
            out = [
                {
                    "class_id": int(i),
                    "class_name": results[0].names.get(int(i), str(int(i))),
                    "confidence": float(probs.data[int(i)]),
                }
                for i in idxs
            ]

        return JSONResponse(content=out)

    except Exception as e:
        log.exception("[YOLOv8-CLS] Inference failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
